# train_lora_conll03.py

import sys
#sys.path.append("/content/dynamo") 

import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# COLAB替代方式：直接添加项目根路径
project_root = "/content/dynamo" 
sys.path.append(project_root)

import json
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType

from utils.train_utils import set_seed, print_trainable_params, freeze_base_model
from trainers.trainer_cls_token import TokenClassificationTrainer
from utils.setting_utils import parse_args, load_config, apply_path_placeholders
from utils.train_utils import set_seed, print_trainable_params, freeze_base_model
from utils.task_map import get_task_info
from model.custom_cls_model import CustomClassificationModel

def load_label_map(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)  # expects {"O": 0, "B-PER": 1, ...}
    
def load_tokenizer_and_model(config, label2id):
    id2label = {v: k for k, v in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(config['backbone_model'])
    
    model = CustomClassificationModel(
        config['backbone_model'],
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )
    return tokenizer, model

def load_datasets(config, tokenizer, label2id):
    task_info = get_task_info(config['task_name'].lower())
    DatasetClass = task_info["dataset_class"]
    extra_args = task_info.get("extra_args", {}).copy()

    # 注入 label2id
    if "label2id" in extra_args and extra_args["label2id"] is None:
        extra_args["label2id"] = label2id

    train_dataset = DatasetClass(
        file_path=config['data']['train_file'],
        tokenizer=tokenizer,
        max_length=config['train']['max_seq_length'],
        **extra_args
    )
    val_dataset = DatasetClass(
        file_path=config['data']['val_file'],
        tokenizer=tokenizer,
        max_length=config['train']['max_seq_length'],
        **extra_args
    )
    return train_dataset, val_dataset

def main(config_path):
    config = load_config(config_path)
    config = apply_path_placeholders(config)
    print(f"\n🚀 Training task: {config['task_name']} started.")
    
    set_seed(config['train']['seed'])

    # Step 1. 读取标签映射 & 加载 tokenizer 和 model
    label2id = load_label_map(config['data']['label2id_file'])
    config['label2id'] = label2id
    tokenizer, base_model = load_tokenizer_and_model(config, label2id)

    # Step 2. 注入 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        bias='none',
        target_modules=config['lora']['target_modules']
    )

    model = get_peft_model(base_model, peft_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    freeze_base_model(model)
    print_trainable_params(model)
    model.to(device)

     # Step 3. 准备数据集与 DataLoader
    train_dataset, val_dataset = load_datasets(config, tokenizer, label2id)
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])

    # Step 4. 进行训练
    config['train']['steps_per_epoch'] = len(train_loader)
    trainer = TokenClassificationTrainer(model, config, device, tokenizer)
    trainer.train(train_loader, val_loader)

    print("\n🚀 TensorBoard started! Run this command to view logs:\n")
    print(f"   tensorboard --logdir={config['output']['log_dir']}")
    print("Then open http://localhost:6006 in your browser.\n")

if __name__ == "__main__":
    try:
        # 在COLAB建CELL：
        # #!python train_lora/train_lora_conll03.py --config configs/single_lora_conll03.yaml
        
        # 👇 若要切换任务，只需改 config 文件名即可：
        # single_lora_conll03.yaml
        if 'google.colab' in sys.modules:
            sys.argv = ['train_lora_conll03.py', '--config', '/content/dynamo/configs/single_lora_conll03.yaml']
        
        args = parse_args()
        main(args.config)
    except Exception as e:
        print(f"[FATAL] Uncaught exception: {e}")
        import traceback
        traceback.print_exc()
