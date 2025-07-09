import numpy as np
import torch
import random
from transformers.optimization import Adafactor
from torch.optim import AdamW

def set_seed(seed):
    random.seed(seed)   # Python 随机性（如 shuffle）
    np.random.seed(seed)    # NumPy 随机性（如随机采样）
    torch.manual_seed(seed) # PyTorch CPU 随机性
    torch.cuda.manual_seed_all(seed) # 	PyTorch GPU 随机性

def freeze_base_model(model):
    # Freeze all original model parameters
    for name, param in model.named_parameters():
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
    