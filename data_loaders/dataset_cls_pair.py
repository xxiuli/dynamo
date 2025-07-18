# dataset_cls_pair.py           # mnli, qqp

# data_loaders/dataset_cls_pair.py

import json
import torch
from torch.utils.data import Dataset

class PairedTextClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)

                # 兼容不同字段
                if "premise" in item and "hypothesis" in item:
                    sent1 = item["premise"]
                    sent2 = item["hypothesis"]
                elif "question1" in item and "question2" in item:
                    sent1 = item["question1"]
                    sent2 = item["question2"]
                else:
                    raise ValueError(f"[ERROR] Unknown paired input keys in item: {item}")

                self.samples.append({
                    "sentence1": sent1,
                    "sentence2": sent2,
                    "label": int(item["label"])
                })

        if not self.samples:
            raise ValueError(f"[ERROR] Empty dataset at: {file_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["sentence1"],
            item["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }

