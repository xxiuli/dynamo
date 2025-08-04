# get_mix_data.py for inference
import os
import yaml
import json
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.task_id_map import get_task2id

# ==== 加载 YAML 配置 ====
def load_config(yaml_path):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config file: {e}")

# ==== 每个任务的数据抽取函数 ====
def extract_text_and_label(example, task_name):
    try:
        if task_name == "sst2":
            return example["sentence"], example["label"]
        elif task_name == "mnli":
            text = f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}"
            return text, example["label"]
        elif task_name == "qqp":
            text = f"Q1: {example['question1']} Q2: {example['question2']}"
            return text, example["label"]
        elif task_name == "squad":
            text = f"Question: {example['question']} Context: {example['context']}"
            return text, example.get("answers", {}).get("text", [""])[0]  # use the first answer
        elif task_name == "xsum":
            return example["document"], example["summary"]
        elif task_name == "agnews":
            return example['text'], example["label"]
        elif task_name == "conll2003" or task_name == "conll03":
            return " ".join(example["tokens"]), example["ner_tags"]
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
            text, label = extract_text_and_label(ex, task_name)
            ex["text"] = text
            ex["task_id"] = task_id
            ex["task_name"] = task_name
            ex["label"] = label
            output_list.append(ex)
            # text = f"[TASK={task_name}] {text}"
            # output_list.append({
            #     "text": text,
            #     "task_id": task_id,
            #     "task_name": task_name,
            #     "label": label
            # })
        except Exception as e:
            print(f"[!] Skipping example {idx} from {task_name} due to error: {e}")

# ==== 主函数 ====
def build_router_dataset(yaml_path, save_dir="data/end2end_mix"):
    os.makedirs(save_dir, exist_ok=True)
    cfg = load_config(yaml_path)
    task_id_map = get_task2id()
    
    seed = cfg.get("sampling", {}).get("seed", 42)
    
    shuffle = cfg.get("sampling", {}).get("shuffle", True)
    
    tasks = cfg["tasks"]

    router_test = []

    for task_name, task_cfg in tasks.items():
        # ✅ 用映射取出 task_id，而不是用 enumerate
        if task_name not in task_id_map:
            print(f"[❌] Task {task_name} not found in task_id_map.json. Skipping...")
            continue
        task_id = task_id_map[task_name]

        try:
            process_task(task_name, task_id, task_cfg, "test", router_test, seed, shuffle)
        except Exception as e:
            print(f"[!!] Error processing task {task_name}: {e}")

        try:
            with open(os.path.join(save_dir, f"data_inference_{task_name}.jsonl"), "w", encoding="utf-8") as f:
                for item in router_test:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception as e:
            raise RuntimeError(f"Failed to write output files: {e}")

    print(f"\n✅ Finished! Saved: {len(router_test)} train samples, {len(router_test)} val samples → {save_dir}/")


# ✅ 使用示例
if __name__ == "__main__":
    build_router_dataset("data/download_configs/test_sampling_single_type.yaml")
