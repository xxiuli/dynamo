import torch
import argparse
from utils.setting_utils import load_config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from router.router_classifier import RouterClassifier
from model.custom_cls_model import CustomClassificationModel
import os
from peft import PeftModel

# ----------------------
# 初始化
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", type=str, required=True, help="Path to this script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()

def load_router(router_cfg):
    # 使用 router.tokenizer 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(router_cfg['tokenizer'])

    # 初始化 Router 模型结构
    model = RouterClassifier.from_pretrained(router_cfg['checkpoint_path'])

    return tokenizer, model

def load_adapter_model(task_cfg, adapter_cfg):
    task_type = task_cfg['task_type'].lower()
    model_dir = task_cfg['adapter_path']

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if task_type == 'classification':
        # ${DRIVE_ROOT}/DynamoRouterCheckpoints/adapter_agnews
        model = CustomClassificationModel.from_pretrained(
            model_dir,
            num_labels=adapter_cfg['num_labels']
            )
        return tokenizer, model

    elif task_type in ['qa', 'summarization']:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.eval()
        return tokenizer, model
    else:
        raise ValueError(f"[ERROR] Unknown task type: {task_type}")

def main():
   args = parse_args()
   config = load_config(args.config)

   # 1. 加载 Router
   router_cfg = config['router']
   router_model, router_tokenizer = load_router(router_cfg)

   # 2. 加载所有 Adapter 模型
   tasks = config['tasks']
   task_models = {}
   for task_name, task_cfg in tasks.items():
       adapter_cfg = load_config(task_cfg['config_path'])
       task_models[task_name] = load_adapter_model(task_cfg, adapter_cfg)
   

if __name__ == '__main__':
    main()