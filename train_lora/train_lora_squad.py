# train_lora_squad.py
import sys
#sys.path.append("/content/dynamo") 

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# COLAB替代方式：直接添加项目根路径
#project_root = "/content/dynamo"  # 或你实际项目根目录
#sys.path.append(project_root)
# train_lora_squad.py

import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from peft import get_peft_model, LoraConfig, TaskType

from utils.train_utils import set_seed, print_trainable_params, freeze_base_model
from data_loaders.dataset_qa import QuestionAnsweringDataset
from trainers.trainer_qa import QuestionAnsweringTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    # 定义命令行运行所需参数，最终： python train_lora_cls.py --config configs/single_lora_agnews.yaml
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
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

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(config['backbone_model'])
        base_model = AutoModelForQuestionAnswering.from_pretrained(config['backbone_model'])
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load model/tokenizer: {e}")

    try:
        peft_config = LoraConfig(
            task_type=TaskType.QUESTION_ANS,
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
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to apply LoRA config: {e}")

    try:
        train_dataset = QuestionAnsweringDataset(
            file_path=config['data']['train_file'],
            tokenizer=tokenizer,
            max_length=config['train']['max_seq_length'],
            doc_stride=config['train'].get('doc_stride', 128)
        )

        val_dataset = QuestionAnsweringDataset(
            file_path=config['data']['val_file'],
            tokenizer=tokenizer,
            max_length=config['train']['max_seq_length'],
            doc_stride=config['train'].get('doc_stride', 128)
        )

        train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])

    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to prepare datasets or dataloaders: {e}")

    config['train']['steps_per_epoch'] = len(train_loader)
    trainer = QuestionAnsweringTrainer(model, config, device, tokenizer)

    try:
        trainer.train(train_loader, val_loader)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise

    print(f"\n🚀 TensorBoard started! Run this command to view logs:\n")
    print(f"   tensorboard --logdir={config['output']['log_dir']}")
    print("Then open http://localhost:6006 in your browser.\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Uncaught exception: {e}")
