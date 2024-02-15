'''
Adapted from https://github.com/huggingface/transformers
'''

from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, __HEAD_MASK_WARNING_MSG
import copy
import os
import warnings
from typing import Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)


class T5ForMultimodalGenerationVSCDCoT(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, patch_size, padding_idx, save_dir,vot_num,alpha):
        super().__init__(config)
        self.model_dim = config.d_model
        self.vot_num=vot_num
        self.alpha=alpha
        self.padding_idx = padding_idx
        self.generate_cot = False
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_num, self.patch_dim = patch_size

        self.image_dense = nn.Linear(self.patch_dim, config.d_model)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size, vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.gate_dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
    def set_conf(self,generate_cot):
        self.generate_cot = generate_cot
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_ids = None,
        r_image_ids = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        r_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        decoder_input_ids_base=decoder_input_ids
        # decoder_attention_mask_base=decoder_attention_mask
        decoder_inputs_embeds_base=decoder_inputs_embeds
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=True,  # 设置这里 output_attentions
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )


        hidden_states_o = encoder_outputs[0]

        image_embedding = self.image_dense(image_ids)
        image_att, _ = self.mha_layer(hidden_states_o, image_embedding, image_embedding)

        merge = torch.cat([hidden_states_o, image_att], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        hidden_states = (1 - gate) * hidden_states_o + gate * image_att
        # return hidden_states[0][189]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        # all_logits = []

        for _ in range(self.vot_num):
            # Decode
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = decoder_outputs[0]
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                sequence_output = sequence_output.to(self.lm_head.weight.device)

            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim**-0.5)

            lm_logits = self.lm_head(sequence_output)
        # softmax = nn.Softmax()
        # v,i = softmax(lm_logits[0][276]).topk(k=5,dim=-1)
        # print(v,i)
        # print(lm_logits.shape)
        # return lm_logits#[0][189]
        
            # all_logits.append(lm_logits)
        #
        # # voting
        # stacked_logits = torch.stack(all_logits, dim=0)
        # mean_logits = torch.mean(stacked_logits, dim=0)
        # stddev_logits = torch.std(stacked_logits, dim=0)
        # weights = 1 / (1 + stddev_logits)
        # weighted_mean_logits = torch.sum(weights * stacked_logits, dim=0) / torch.sum(weights, dim=0)
        # alpha = self.alpha
        # lm_logits = alpha * mean_logits + (1 - alpha) * weighted_mean_logits
        if r_image_ids is not None and self.generate_cot:
            with torch.no_grad():
                image_embedding = self.image_dense(r_image_ids)
                image_att, _ = self.mha_layer(hidden_states_o, image_embedding, image_embedding)

                merge = torch.cat([hidden_states_o, image_att], dim=-1)
                gate = self.sigmoid(self.gate_dense(merge))
                hidden_states_r = (1 - gate) * hidden_states_o + gate * image_att

            if r_labels is not None:
                # get decoder inputs from shifting lm labels to the right
                decoder_input_ids_r = self._shift_right(labels)#(r_labels)
            else:
                decoder_input_ids_r = self._shift_right(labels)
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)
                hidden_states_r = hidden_states_r.to(self.decoder.first_device)
                if decoder_input_ids_r is not None:
                    decoder_input_ids_r = decoder_input_ids_r.to(self.decoder.first_device)
            # Decode
            with torch.no_grad():
                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids_r,    #需要额外的参数
                    attention_mask=decoder_attention_mask,
                    inputs_embeds=decoder_inputs_embeds,
                    past_key_values=past_key_values,
                    encoder_hidden_states=hidden_states_r,
                    encoder_attention_mask=attention_mask,
                    head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            sequence_output = decoder_outputs[0]
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                sequence_output = sequence_output.to(self.lm_head.weight.device)

            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim ** -0.5)

            r_lm_logits = self.lm_head(sequence_output)


        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # print(loss,end=',')
            if r_image_ids is not None and self.generate_cot:
                # TODO: 计算对比解码logits，蒸馏
                alpha = 1
                beta = 0.1
                T = 2
                _lambda = 0.3
                # lm_logits_min = lm_logits.view(-1).min(dim=-1).values
                cutoff = beta * lm_logits.max(dim=-1, keepdim=True).values #torch.log(torch.tensor(beta)) + lm_logits.max(dim=-1, keepdim=True).values
                # cutval = lm_logits.view(-1).max(dim=-1).values
                diffs =  lm_logits - r_lm_logits
                cutval = diffs.view(-1).min(dim=-1).values

                # 过滤cutoff
                cd_logits = lm_logits + alpha * diffs.masked_fill(lm_logits < cutoff, cutval)
                # cd_logits = cd_logits.masked_fill(cd_logits < lm_logits_min, lm_logits_min)
                a=torch.nonzero(torch.isnan(cd_logits)==True).size()
                # print(a)
                kl = F.kl_div(
                    F.log_softmax(F.sigmoid(lm_logits) / T, dim=-1),
                    F.softmax(F.sigmoid(cd_logits) / T, dim=-1),
                    reduction='batchmean')
                # kl = F.kl_div(
                #     F.log_softmax((lm_logits) / T, dim=-1),
                #     F.softmax((cd_logits) / T, dim=-1),
                #     reduction='batchmean')
                loss = (1 - _lambda) * loss + _lambda * (T ** 2) * kl
                if a[0]>0:
                    print(loss,kl,cd_logits.view(-1).min(dim=-1).values,diffs.view(-1).min(dim=-1).values)
                    exit()
                # print(loss)


        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )