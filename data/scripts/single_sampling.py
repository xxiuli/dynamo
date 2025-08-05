import yaml
import os
import argparse
from datetime import datetime
from datasets import load_dataset

timestamp = datetime.now().strftime("%m%d%Y_%H%M")

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_PATH, 'single_lora_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_config(config_doc):
    with open(config_doc, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# python single_sampling.py --config configs/single_sampling_tasks.yaml --task sst2 mnli squad  --output_dir data/raw/  
def set_parse_args():
    parser = argparse.ArgumentParser(description="Single Task Sampling")
    parser.add_argument('--config_doc', type=str, required=True, help='Path to sampling config.yaml')
    parser.add_argument('--task', type=str, nargs='+', required=True, help="Task name (e.g., sst2, mnli)")
    return parser.parse_args()

def get_samples(seed, shuffle, task_dic):
    dataset_name = task_dic['dataset_name'] #'glue'
    subset = task_dic['subset'] #'sst2'
    train_size = task_dic['train_samples'] #5000
    val_size = task_dic['val_samples'] #1000
    train_split = task_dic['train_split']
    val_split = task_dic['val_split']

    file_prefix = f'{dataset_name}_{subset}' if subset else f'{dataset_name}'

    # download 
    print(f"\nLoading dataset: {dataset_name} {f'({subset})' if subset else ''}")
    try:
        if subset:
            data = load_dataset(dataset_name, subset,trust_remote_code=True)
        else:
            data = load_dataset(dataset_name, trust_remote_code=True)
        print(f"\nSucssful load: {dataset_name} {f'({subset})' if subset else ''}")
    except Exception as e:
        print(f"Error while Loading dataset: {e}")
        return

    try:
        # 1. 获取训练数据
        train_data = data[train_split]

        # 2. 获取验证数据（如果有）
        if val_split is not None:
            val_data = data[val_split]
        else:
            val_data = None

        # 3. 打乱数据（有条件）
        if shuffle:
            train_data = train_data.shuffle(seed=seed)
            if val_data is not None:
                val_data = val_data.shuffle(seed=seed)

        # 4. 采样训练集
        train_samples = train_data if train_size == -1 else train_data.select(range(min(train_size, len(train_data))))
        train_path = os.path.join(OUTPUT_DIR, f'{file_prefix}_{train_split}_{timestamp}.json')
        train_samples.to_json(train_path, orient="records", lines=True, force_ascii=False)
        print(f"[✓] Saved train samples to {train_path}")

        # 5. 采样验证集（可选）
        if val_data is not None:
            val_samples = val_data if val_size == -1 else val_data.select(range(min(val_size, len(val_data))))
            val_split = 'val' if val_split=="train" else val_split
            val_path = os.path.join(OUTPUT_DIR, f'{file_prefix}_{val_split}_{timestamp}.json')
            val_samples.to_json(val_path, orient="records", lines=True, force_ascii=False)
            print(f"[✓] Saved val samples to {val_path}")
        else:
            print(f"[INFO] Skipping val sampling for {file_prefix} (no val_split)")

    except Exception as e:
        print(f"Error while Loading dataset: {e}")
    print()
    
def main():
    args = set_parse_args() #[--config_doc, --task, --output_dir]
    # print(args)
    configs = load_config(args.config_doc) # configs/single_sampling.yaml
    # print(config)

    seed = configs['sampling']['seed'] #42
    shuffle = configs['sampling']['shuffle'] #true

    count = 1
    taskTotal = len(args.task)
    for task in args.task:
        if task not in configs['tasks']:
            raise ValueError(f"Task '{task}' not found in config file")
        
        print(f'Task {count}/{taskTotal}')
        task_dic= configs['tasks'][task] #'sst2'

        get_samples(seed, shuffle, task_dic)
        print(f'Task {count}/{taskTotal} is complete.')
        count +=1


if __name__ == '__main__':
    main()