# train_lora_cls.py ->agnews、sst2、mnli、qqp

import sys
#sys.path.append("/content/dynamo") 

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# COLAB替代方式：直接添加项目根路径
#project_root = "/content/dynamo"  # 或你实际项目根目录
#sys.path.append(project_root)
# train_lora_squad.py

import yaml
import torch
import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType

from utils.train_utils import set_seed, print_trainable_params, freeze_base_model
#text
from data_loaders.dataset_cls_single import SingleTextClassificationDataset
from data_loaders.dataset_cls_pair import PairTextClassificationDataset
from trainers.trainer_cls_single import SingleClassificationTrainer


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
        raise ValueError(f"[ERROR] YAML format error: {e}")


def get_dataset_class(task_name):
    if task_name in ['agnews', 'sst2']:
        return SingleTextClassificationDataset
    elif task_name in ['mnli', 'qqp']:
        return PairTextClassificationDataset
    else:
        raise ValueError(f"[ERROR] Unsupported classification task: {task_name}")


def main():
    try:
        args = parse_args()
        config = load_config(args.config)
        set_seed(config['train']['seed'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        task_name = config['task_name'].lower()
        DatasetClass = get_dataset_class(task_name)

        try:
            tokenizer = AutoTokenizer.from_pretrained(config['backbone_model'])
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config['backbone_model'],
                num_labels=config['num_labels']
            )
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load tokenizer or model: {e}")

        try:
            train_dataset = DatasetClass(
                file_path=config['data']['train_file'],
                tokenizer=tokenizer,
                max_seq_len=config['train']['max_seq_length']
            )
            val_dataset = DatasetClass(
                file_path=config['data']['val_file'],
                tokenizer=tokenizer,
                max_seq_len=config['train']['max_seq_length']
            )
        except Exception as e:
            raise RuntimeError(f"[ERROR] Dataset loading failed: {e}")

        train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])

        try:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=config['lora']['r'],
                lora_alpha=config['lora']['alpha'],
                lora_dropout=config['lora']['dropout'],
                bias='none',
                target_modules=config['lora']['target_modules']
            )
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to apply LoRA config: {e}")

        try:
            model = get_peft_model(base_model, peft_config)
            model.to(device)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to inject LoRA: {e}")

        freeze_base_model(model)
        print_trainable_params(model)

        config['train']['steps_per_epoch'] = len(train_loader)

        trainer = SingleClassificationTrainer(model, config, device, tokenizer)
        try:
            trainer.train(train_loader, val_loader)
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            raise

        print("\n🚀 TensorBoard started! Run this command to view logs:\n")
        print(f"   tensorboard --logdir={config['output']['log_dir']}")
        print("Then open http://localhost:6006 in your browser.\n")

    except Exception as e:
        print(f"[FATAL] Uncaught exception in main: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
