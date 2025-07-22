import torch
import argparse
from utils.setting_utils import load_config
from transformers import AutoTokenizer
from router.router_classifier import RouterClassifier
from model.custom_cls_model import CustomClassificationModel

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
    tokenizer = AutoTokenizer.from_pretrained(router_cfg['tokenizer'])

    model = RouterClassifier(
        hidden_size=router_cfg['hidden_size'],
        num_task=router_cfg['num_task'],
        temperature=router_cfg['temperature']
    )

    model.load_state_dict(torch.load(router_cfg['checkpooint_path'], map_location='cpu'))
    model.eval()

    return tokenizer, model

def load_adapter_model(task_config, adapter_cfg):
    tokenizer = AutoTokenizer.from_pretrained(task_config['tokenizer'])

    if task_config['task_type'] == 'Classification':
        model = CustomClassificationModel(
            backbone_name=adapter_cfg['backbone_model'],
            num_labels=adapter_cfg['num_labels']
        )

def main():
   args = parse_args()
   config = load_config(args.config)

   # 1. 加载 Router
   router_cfg = config['router']
   router_model, router_tokenizer = load_router(router_cfg)

   # 2. 加载所有 Adapter 模型
   tasks = config['tasks']
   task_model = {}
   for task_name, task_cfg in tasks.items():
       adapter_cfg = load_config(task_cfg['config_path'])
       task_model[task_name] = load_adapter_model(task_cfg, adapter_cfg)
   

if __name__ == '__main__':
    main()