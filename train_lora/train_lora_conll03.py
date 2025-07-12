# train_lora_conll03.py
import sys
#sys.path.append("/content/dynamo") 

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# COLAB替代方式：直接添加项目根路径
#project_root = "/content/dynamo"  # 或你实际项目根目录
#sys.path.append(project_root)
# train_lora_squad.py

import yaml
import json
import torch
import argparse

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType

from utils.train_utils import set_seed, print_trainable_params, freeze_base_model
from data_loaders.dataset_cls_token import TokenClassificationDataset
from trainers.trainer_cls_token import TokenClassificationTrainer

# python train_lora_conll03.py --config configs/single_lora_conll03.yaml
# CONFIG_PATH = "configs/single_lora_conll03.yaml"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_label_map(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)  # expects {"O": 0, "B-PER": 1, ...}

def main():
    args = parse_args()
    config = load_config(args.config)
    # config = load_config(CONFIG_PATH)
    set_seed(config['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config['backbone_model'])
        label2id = load_label_map(config['data']['label2id_file'])
        id2label = {v: k for k, v in label2id.items()}
        config['label2id'] = label2id
        
        base_model = AutoModelForTokenClassification.from_pretrained(
            config['backbone_model'],
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load tokenizer/model or label map: {e}")

    train_dataset = TokenClassificationDataset(
        file_path=config['data']['train_file'],
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=config['train']['max_seq_length']
    )

    val_dataset = TokenClassificationDataset(
        file_path=config['data']['val_file'],
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=config['train']['max_seq_length']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])

    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        bias='none',
        target_modules=config['lora']['target_modules']
    )

    model = get_peft_model(base_model, peft_config)
    model.to(device)

    freeze_base_model(model)
    print_trainable_params(model)

    config['train']['steps_per_epoch'] = len(train_loader)

    trainer = TokenClassificationTrainer(model, config, device, tokenizer)

    try:
        trainer.train(train_loader, val_loader)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise

    print("\n🚀 TensorBoard started! Run this command to view logs:\n")
    print(f"   tensorboard --logdir={config['output']['log_dir']}")
    print("Then open http://localhost:6006 in your browser.\n")

if __name__ == "__main__":
    main()
