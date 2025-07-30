import os
import yaml
import json
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def extract_text(example, task_name):
    if task_name == "sst2":
        return example["sentence"]
    elif task_name == "mnli":
        return f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}"
    elif task_name == "qqp":
        return f"Q1: {example['question1']} Q2: {example['question2']}"
    elif task_name == "squad":
        return f"Question: {example['question']} Context: {example['context']}"
    elif task_name == "xsum":
        return example["document"]
    elif task_name == "agnews":
        return example["text"]
    elif task_name == "conll2003" or task_name == "conll03":
        return " ".join(example["tokens"])
    else:
        raise NotImplementedError(f"Task {task_name} not supported.")

def sample_router_training_data(yaml_path, output_path):
    cfg = load_config(yaml_path)
    seed = cfg.get("sampling", {}).get("seed", 42)
    shuffle = cfg.get("sampling", {}).get("shuffle", True)
    tokenizer_name = cfg.get("sampling", {}).get("tokenizer", "bert-base-uncased")
    min_tokens = cfg.get("sampling", {}).get("min_tokens", 10)
    max_tokens = cfg.get("sampling", {}).get("max_tokens", 512)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    task2id = {task_name: idx for idx, task_name in enumerate(cfg["tasks"].keys())}

    samples = []

    for task_name, task_cfg in cfg["tasks"].items():
        print(f"\n📦 Sampling from: {task_name}")
        dataset_name = task_cfg["dataset_name"]
        subset = task_cfg.get("subset")
        split = task_cfg.get("train_split", "train")
        num_samples = task_cfg.get("train_samples", 100)

        ds = load_dataset(dataset_name, subset, split=split)
        if shuffle:
            ds = ds.shuffle(seed=seed)

        count = 0
        for example in tqdm(ds, desc=task_name):
            if count >= num_samples:
                break
            try:
                text = extract_text(example, task_name)
                tokenized = tokenizer(text, truncation=False)
                num_tokens = len(tokenized["input_ids"])
                if min_tokens <= num_tokens <= max_tokens:
                    samples.append({
                        "text": text,
                        "task_id": task2id[task_name],
                        "task_name": task_name
                    })
                    count += 1
            except Exception as e:
                print(f"[!] Skipped sample due to error: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(samples)} Router samples to {output_path}")

# ✅ 使用示例
if __name__ == "__main__":
    out_path = r"C:\Users\xiuxiuli.SSNC-CORP\Desktop\learn\567ML\dynamo\data\end2end_mix\testset.json"
    sample_router_training_data("data/download_configs/test_sampling.yaml", out_path)
