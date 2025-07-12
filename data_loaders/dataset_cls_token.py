# dataset_cls_token.py -> conll03
import json
import torch
from torch.utils.data import Dataset

class TokenClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.samples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = item['tokens']
        labels = item['ner_tags']  # 标签为 BIO 格式，如 ["B-PER", "I-PER", "O", ...]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,  # 用于对齐 label
            return_tensors="pt"
        )

        labels_aligned = [-100] * self.max_length
        word_ids = encoding.word_ids(batch_index=0)
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx < len(labels):
                labels_aligned[i] = self.label2id[labels[word_idx]]

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels_aligned, dtype=torch.long)
        }
