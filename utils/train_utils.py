# train_utils.py
import os
import numpy as np
import torch
import random
from transformers.optimization import Adafactor
from torch.optim import AdamW
import yaml

class EarlyStop:
    def __init__(self, patience=3, mode='min'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            print(f"[EarlyStop] Init best_score = {score:.4f}")
        elif (self.mode =='min' and score >= self.best_score) or (self.mode == 'max' and score <= self.best_score):
            self.counter += 1
            print(f"[EarlyStop] No improvement (score={score:.4f}), counter={self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(f"[EarlyStop] Improved from {self.best_score:.4f} → {score:.4f}")
            self.best_score = score
            self.counter = 0

def set_seed(seed):
    random.seed(seed)   # Python 随机性（如 shuffle）
    np.random.seed(seed)    # NumPy 随机性（如随机采样）
    torch.manual_seed(seed) # PyTorch CPU 随机性
    torch.cuda.manual_seed_all(seed) # 	PyTorch GPU 随机性

def freeze_base_model(model):
    # Freeze all original model parameters
    for name, param in model.named_parameters():
        # 只有插入的 LoRA 模块（通常名字中包含 lora_）和（可选的）任务分类头 classifier 会参与训练
        if 'lora_' not in name and 'classifier' not in name:
            param.requires_grad = False

def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable params:{trainable} / {total} ({trainable/total:.2%})')

def get_optimizer(name, params, lr):
    name = name.lower()
    if name == 'adamw':
        return AdamW(params, lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    elif name == "adafactor":
        return Adafactor(params, scale_parameter=True, relative_step=True, warmup_init=True)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    
#  2. save_model 工具函数（建议传 trainer 对象）
def save_model(trainer, final=False, epoch=None):
    try:
        task_name = trainer.config['task_name']
        path = os.path.join(
            trainer.save_dir, 
            f'adapter_{task_name}' if final else f'checkpoint_epoch{epoch}'
        )
        os.makedirs(path, exist_ok=True)

        # ✅ 如果模型有 HuggingFace config（例如 AutoModel / PeftModel / CustomClassificationModel）
        if hasattr(trainer.model, 'config') and hasattr(trainer.model.config, 'to_json_string'):
            config_path = os.path.join(path, 'config.json')
            with open(config_path , 'w') as f:
                f.write(trainer.model.config.to_json_string())
            print(f"[INFO] HuggingFace config saved to {config_path}")
        else:
            print(f"[INFO] Skipped saving HuggingFace config (no .config or not serializable)")

        # ✅ 模型保存（支持 HuggingFace、LoRA、RouterClassifier）
        if hasattr(trainer.model, "save_pretrained"):
            trainer.model.save_pretrained(path)
            print(f"[INFO] Model saved using save_pretrained() to {path}")
        else:
            # 兜底方案：保存权重
            model_path = os.path.join(path, "pytorch_model.bin")
            torch.save(trainer.model.state_dict(), model_path)
            print(f"[INFO] Model weights saved to {model_path}")

        # ✅ 保存自定义 Head（如果有）
        if hasattr(trainer.model, "head"):
            head_path = os.path.join(path, "head.pth")
            torch.save(trainer.model.head.state_dict(), head_path)
            print(f"[INFO] Head weights saved to {head_path}")
        else:
            print(f"[INFO] No custom head to save")

        # ✅ 保存 tokenizer（如果有）
        if trainer.tokenizer and hasattr(trainer.tokenizer, "save_pretrained"):
            trainer.tokenizer.save_pretrained(path)
            print(f"[INFO] Tokenizer saved to {path}")
        else:
            print(f"[INFO] No tokenizer saved")

    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")
    