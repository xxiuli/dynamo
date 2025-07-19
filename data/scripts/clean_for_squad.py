# data/scripts/clean_for_squad.py
# data/single_lora_data/clean_for_squad.py

from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
max_length = 384

input_file = "data/single_lora_data/squad_validation_07192025_0549.json"  # 输入原始文件
output_file = "data/single_lora_data/squad_val_cleaned_file.jsonl"  # 输出清洗后文件

count_total, count_kept, count_bad = 0, 0, 0

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        count_total += 1
        item = json.loads(line.strip())
        try:
            question = item["question"]
            context = item["context"]
             # ✅ tokenizer 不抛异常就表示通过
            _ = tokenizer(
                question,
                context,
                max_length=max_length,
                truncation=True,  # 一定要用True
                padding="max_length",
                stride=128,
                return_offsets_mapping=True,
                return_tensors="pt"
            )
            fout.write(json.dumps(item) + "\n")
            count_kept += 1
        except Exception as e:
            count_bad += 1
            continue

print(f"[INFO] Total={count_total}, Kept={count_kept}, Removed={count_bad}")
