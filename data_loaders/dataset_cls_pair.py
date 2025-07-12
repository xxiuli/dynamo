# dataset_cls_pair.py           # mnli, qqp

import json
import torch
from torch.utils.data import Dataset

class PairTextClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if 'premise' in item and 'hypothesis' in item:
                    text1 = item['premise']
                    text2 = item['hypothesis']
                elif 'question1' in item and 'question2' in item:
                    text1 = item['question1']
                    text2 = item['question2']
                else:
                    raise ValueError("Invalid sample format: missing expected pair fields")
                label = item['label']
                self.samples.append((text1, text2, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text1, text2, label = self.samples[idx]
        encoding = self.tokenizer(
            text1,
            text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
