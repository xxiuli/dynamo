from torch.utils.data import Dataset
import json
import torch
import os

class RouterDataset(Dataset):
    def __init__(self, json_path, tokenizer,task_id_map, max_length=512):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"[RouterDataset] File not found: {json_path}")
        
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_id_map = task_id_map
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        task_name = item["task_name"]
        label = self.task_id_map[task_name]

        encoded = self.tokenizer(
            item["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
            )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "task_name": torch.tensor(task_name),
            "label": torch.tensor(label)  # 分类器 label 就是 task_id
        }
