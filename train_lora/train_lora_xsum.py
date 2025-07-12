# train_lora_xsum.py
import sys
#sys.path.append("/content/dynamo") 

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# COLAB替代方式：直接添加项目根路径
#project_root = "/content/dynamo"  # 或你实际项目根目录
#sys.path.append(project_root)

import json
import yaml
import torch
import argparse

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType

from utils.train_utils import set_seed, freeze_base_model, print_trainable_params
from data_loaders.dataset_summarization import SummarizationDataset
from trainers.trainer_summarize import SummarizationTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    return parser.parse_args()


def load_config(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] Config file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"[ERROR] Invalid YAML format: {e}")


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['train']['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config['backbone_model'])
        base_model = AutoModelForSeq2SeqLM.from_pretrained(config['backbone_model'])
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load tokenizer or base model: {e}")

    try:
        train_dataset = SummarizationDataset(
            file_path=config['data']['train_file'],
            tokenizer=tokenizer,
            max_source_length=config['train']['max_source_length'],
            max_target_length=config['train']['max_target_length']
        )

        val_dataset = SummarizationDataset(
            file_path=config['data']['val_file'],
            tokenizer=tokenizer,
            max_source_length=config['train']['max_source_length'],
            max_target_length=config['train']['max_target_length']
        )

        train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to prepare datasets or dataloaders: {e}")

    try:
        if "bart" in config['backbone_model'].lower():
            target_modules = ["q_proj", "v_proj"]
        elif "roberta" in config['backbone_model'].lower():
            target_modules = ["query", "value"]

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
            bias='none',
            target_modules=target_modules
        )

        model = get_peft_model(base_model, peft_config)
        model.to(device)

        freeze_base_model(model)
        print_trainable_params(model)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to apply LoRA config: {e}")

    config['train']['steps_per_epoch'] = len(train_loader)

    trainer = SummarizationTrainer(model, config, device, tokenizer)

    try:
        trainer.train(train_loader, val_loader)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise

    print("\n✅ Training finished. To visualize logs, run:\n")
    print(f"   tensorboard --logdir={config['output']['log_dir']}")
    print("Then open http://localhost:6006 in your browser.\n")


if __name__ == "__main__":
    main()

