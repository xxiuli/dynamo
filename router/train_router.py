# router/train_router.py
import sys
#sys.path.append("/content/dynamo") 

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# COLAB替代方式：直接添加项目根路径
#project_root = "/content/dynamo"  # 或你实际项目根目录
#sys.path.append(project_root)
import yaml
import torch
from router_classifier import RouterClassifier
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer, get_scheduler
from head_manager import get_head
from torch.utils.data import DataLoader
from data_loaders.dataset_router import RouterDataset
from utils.setting_utils import parse_args, apply_path_placeholders
from utils.train_utils import set_seed
from trainers.trainer_router import RouterTrainer

def load_router_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f :
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[Error] while loading router config: {e}")

def load_adapter(adapter_path, backbone_name):
    base_model = AutoModel.from_pretrained(backbone_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model


def main():
    config = load_router_config("configs/router.yaml")
    config = apply_path_placeholders(config)
    print(f"\n🚀 Training task: {config['task_name']} started.")

    set_seed(config['training']['seed'])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['backbone'])

    model = RouterClassifier(
        num_task=config['num_labels'],
        temperature=config['router']['temperature']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load dataset
    train_dataset = RouterDataset(config['data']['train_file'], tokenizer, config['training']['max_seq_length'])
    val_dataset = RouterDataset(config['data']['val_file'], tokenizer, config['training']['max_seq_length'])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])

    # Prepare trainer
    config['training']['steps_per_epoch'] = len(train_loader)
    trainer = RouterTrainer(model, config, device)

    # Train
    trainer.train(train_loader, val_loader)

    print(f"\n✅ Training finished. Run this to view logs:\n")
    print(f"   tensorboard --logdir={config['output']['log_dir']}\n")

if __name__ == "__main__":
    try:
        # !python router/train_router.py --config configs/router.yaml
        import sys
        # 仅当在 Colab 或 Jupyter 环境下运行时 mock sys.argv
        if 'google.colab' in sys.modules:
            sys.argv = ['train_router.py', '--config', '/content/dynamo/configs/router.yaml']
        
        args = parse_args()
        main()
    except Exception as e:
        print(f"[FATAL] Uncaught exception: {e}")
        import traceback
        traceback.print_exc()