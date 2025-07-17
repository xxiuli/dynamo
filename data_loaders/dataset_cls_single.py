# data_loaders/dataset_cls_single.py -> agnews, sst2

import json
import torch
from torch.utils.data import Dataset

class SingleTextClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append({
                    'text': item['text'] if 'text' in item else item['sentence'],
                    'label': int(item['label'])
                })

        if not self.samples:
            raise ValueError(f"[ERROR] Dataset is empty: {file_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_len,
            return_tensors='pt'
        )
        
        raw_label = item['label']
        try:
            label = int(raw_label)  # 保证是 int (需要计算LOSS的时候不是字符串)
        except ValueError:
            raise ValueError(f"Invalid label: {item['label']} (type: {type(item['label'])})")
        
         # 🔍 打印调试信息
        # label_tensor = torch.tensor(label, dtype=torch.long)
        # print(f"[DEBUG] Sample idx={idx}, raw_label={raw_label}, raw_type={type(raw_label)}")
        # print(f"[DEBUG] label_tensor={label_tensor}, tensor_dtype={label_tensor.dtype}, tensor_type={type(label_tensor)}")

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor([label], dtype=torch.long)  # ⚠️ Wrap label in list to make it rank-1 tensor
        }
