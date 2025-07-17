import os
import yaml
import json
from datasets import load_dataset, Dataset
from tqdm import tqdm

# ==== 加载 YAML 配置 ====
def load_config(yaml_path):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config file: {e}")

# ==== 每个任务的数据抽取函数 ====
def extract_text(example, task_name):
    try:
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
            return example['text']
        elif task_name == "conll2003" or task_name == "conll03":
            return " ".join(example["tokens"])
        else:
            raise NotImplementedError(f"Task {task_name} not supported.")
    except KeyError as ke:
        raise ValueError(f"Missing expected key for task {task_name}: {ke}")

# ==== 执行采样与写入 ====
def process_task(task_name, task_id, task_cfg, split_name, output_list, seed, shuffle):
    name = task_cfg["dataset_name"]
    subset = task_cfg.get("subset")
    split = task_cfg[split_name + "_split"]
    max_samples = task_cfg[split_name + "_samples"]

    print(f"\n📦 Processing [{task_name}] ({split_name})...")
    try:
        ds = load_dataset(name, subset, split=split)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {name} ({task_name}): {e}")

    if shuffle:
        try:
            ds = ds.shuffle(seed=seed)
        except Exception as e:
            print(f"[!] Shuffle failed for {task_name}: {e}")

    if max_samples != -1:
        ds = ds.select(range(min(max_samples, len(ds))))

    for idx, ex in enumerate(tqdm(ds, desc=f"{task_name}-{split_name}")):
        try:
            text = extract_text(ex, task_name)
            output_list.append({
                "text": text,
                "task_id": task_id,
                "task_name": task_name
            })
        except Exception as e:
            print(f"[!] Skipping example {idx} from {task_name} due to error: {e}")

# ==== 主函数 ====
def build_router_dataset(yaml_path, save_dir="router_data"):
    os.makedirs(save_dir, exist_ok=True)
    cfg = load_config(yaml_path)
    
    seed = cfg.get("sampling", {}).get("seed", 42)
    
    shuffle = cfg.get("sampling", {}).get("shuffle", True)
    
    tasks = cfg["tasks"]

    router_train, router_val = [], []

    for task_id, (task_name, task_cfg) in enumerate(tasks.items()):
        try:
            process_task(task_name, task_id, task_cfg, "train", router_train, seed, shuffle)
            process_task(task_name, task_id, task_cfg, "val", router_val, seed, shuffle)
        except Exception as e:
            print(f"[!!] Error processing task {task_name}: {e}")

    try:
        with open(os.path.join(save_dir, "router_train.jsonl"), "w", encoding="utf-8") as f:
            for item in router_train:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(os.path.join(save_dir, "router_val.jsonl"), "w", encoding="utf-8") as f:
            for item in router_val:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as e:
        raise RuntimeError(f"Failed to write output files: {e}")

    print(f"\n✅ Finished! Saved: {len(router_train)} train samples, {len(router_val)} val samples → {save_dir}/")

# ==== 启动 ====
if __name__ == "__main__":
    build_router_dataset("data/download_configs/router_sampling.yaml")
