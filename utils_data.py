import json
import numpy as np
import os.path as osp
import os
import torch
import random
from torch.utils.data import Dataset
from utils_prompt import *
from tqdm import trange
# remove_ids = ['1262', '2477', '3405', '3839', '3906', '4142', '5061', '8001', '12146', '13037', '13487', '14340', '16404', '16857', '16965']#, '18903']
img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
}

def load_data_std(args):
    problems = json.load(open(osp.join(args.data_root, 'scienceqa/problems.json')))
    pid_splits = json.load(open(osp.join(args.data_root, 'scienceqa/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids

def load_data_img(args):
    problems = json.load(open(osp.join(args.data_root, 'scienceqa/problems.json')))
    pid_splits = json.load(open(osp.join(args.data_root, 'scienceqa/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]
    name_maps = json.load(open(osp.join(args.data_root, 'vision_features/name_map.json')))

    # check
    if args.img_type == "resnet":
        image_features = np.load(osp.join(args.data_root, 'vision_features/resnet.npy'))
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif args.img_type == "clip":
        image_features = np.load(osp.join(args.data_root, 'vision_features/clip.npy'))
    elif args.img_type == "detr":
        image_features = np.load(osp.join(args.data_root, 'vision_features/detr.npy'))
    else:
        image_features = np.load(osp.join(args.data_root, 'vision_features/detr.npy'))
    print("img_features size: ", image_features.shape)

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    # for i in remove_ids:
    #     if i in train_qids:
    #         train_qids.remove(i)
    train_rqids = None
    if args.prompt_format == 'QCM-LE':
        file_path = "data/train_rqids.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                train_rqids = json.load(file)
        else:
            train_rqids = [[]]*len(train_qids)
            # ToDo: 查找每个qid index对应的相同问题不同答案的qid index。结果保存。
            for i in trange(len(train_qids)):
                for j in range(i+1,len(train_qids)):
                    instance_i = problems[train_qids[i]]
                    instance_j = problems[train_qids[j]]
                    if instance_i["question"] == instance_j["question"]:
                        if instance_i["choices"][instance_i["answer"]] == instance_j["choices"][instance_j["answer"]]:
                            train_rqids[i].append(j)
                            train_rqids[j].append(i)
            # # 保存 train_rqids 到文件
            # with open(file_path, "w") as file:
            #     json.dump(train_rqids, file)

    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")
    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids, name_maps, image_features,train_rqids


class ScienceQADatasetStd(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataset,  tokenizer, source_len, target_len, args, test_le=None
    ):
        self.tokenizer = tokenizer
        self.data = dataset
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair(dataset, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
        }


class ScienceQADatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, name_maps, tokenizer, source_len, target_len, args, image_features,rqids=None, test_le=None
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.rqids = rqids
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.image_ids = []
        shape = img_shape[args.img_type]
        self.no_img_ids = np.zeros(shape)
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)
            if str(qid) in name_maps:
                i_vectors = image_features[int(name_maps[str(qid)])]
                self.image_ids.append(i_vectors)
            else:
                self.image_ids.append(np.zeros(shape))
    
    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        
        image_ids = self.image_ids[index]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()#.tolist()
        image_ids = torch.tensor(image_ids).squeeze()
        # print(self.rqids)
        if self.rqids is None:
            return {
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "image_ids": image_ids,
                "labels": target_ids,
            }
        else:
            # ToDo: 采样r_image_ids
            r_image_ids = self.no_img_ids
            r_target_ids = None
            if len(self.rqids[index]) != 0:
                r_index = random.sample(self.rqids[index],1)[0]
                r_image_ids = self.image_ids[r_index]
                r_target_text = str(self.target_text[r_index])
                r_target_text = " ".join(r_target_text.split())
                r_target = self.tokenizer.batch_encode_plus(
                    [r_target_text],
                    max_length=self.summ_len,
                    pad_to_max_length=True,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                r_target_ids = r_target["input_ids"].squeeze()

            r_image_ids = torch.tensor(r_image_ids).squeeze()
            return {
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "image_ids": image_ids,
                "r_image_ids": r_image_ids,
                'r_labels': r_target_ids,
                "labels": target_ids,
            }