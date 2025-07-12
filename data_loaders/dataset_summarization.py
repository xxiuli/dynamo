# data_loaders/dataset_summarization.py -> xsum

import json
import torch
from torch.utils.data import Dataset

class SummarizationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_source_length=512, max_target_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        source_text = item['document']
        target_text = item['summary']

        source_encodings = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_encodings = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = target_encodings['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100  # 为了不计算PAD loss

        return {
            'input_ids': source_encodings['input_ids'].squeeze(0),
            'attention_mask': source_encodings['attention_mask'].squeeze(0),
            'labels': labels
        }
