# data_loaders/dataset_qa.py -> squad

import json
import torch
from torch.utils.data import Dataset

class QuestionAnsweringDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=384, doc_stride=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        question = item['question']
        context = item['context']
        if not item['answers']['text']:
            answer_text = ""
            answer_start = 0
        else:
            try:
                answer_text = item['answers']['text'][0]
                answer_start = item['answers']['answer_start'][0]
            except IndexError:
                raise ValueError(f"[ERROR] Missing answer for item: {item['id']}")


        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation="only_second",
            padding="max_length",
            stride=self.doc_stride,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # 计算 answer span 的 start/end token index
        offset_mapping = encoding["offset_mapping"][0]
        start_char = answer_start
        end_char = answer_start + len(answer_text)

        sequence_ids = encoding.sequence_ids(0)
        start_idx = end_idx = 0

        for i, (offset, seq_id) in enumerate(zip(offset_mapping, sequence_ids)):
            if seq_id != 1:
                continue  # 只考虑 context 部分
            if offset[0] <= start_char and offset[1] > start_char:
                start_idx = i
            if offset[0] < end_char and offset[1] >= end_char:
                end_idx = i

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "start_positions": torch.tensor(start_idx, dtype=torch.long),
            "end_positions": torch.tensor(end_idx, dtype=torch.long)
        }
