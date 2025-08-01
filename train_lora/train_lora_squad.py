# train_lora_squad.py
import sys
#sys.path.append("/content/dynamo") 

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# COLAB替代方式：直接添加项目根路径
#project_root = "/content/dynamo"  # 或你实际项目根目录
#sys.path.append(project_root)

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from trainers.trainer_qa import QuestionAnsweringTrainer
from utils.setting_utils import parse_args, load_config, apply_path_placeholders
from utils.train_utils import set_seed, print_trainable_params, freeze_base_model
from utils.task_map import get_task_info
from model.custom_cls_model import CustomClassificationModel

def load_tokenizer_and_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config['backbone_model'])

    base_model = AutoModelForQuestionAnswering.from_pretrained(
        config['backbone_model'], 
        num_labels=config['num_labels'],
        ignore_mismatched_sizes=True
        )
    return tokenizer, base_model

def load_datasets(config, tokenizer):
    task_info = get_task_info(config['task_name'].lower())
    DatasetClass = task_info["dataset_class"]
    extra_args = task_info.get("extra_args", {}).copy()

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

    # Step 1. 加载 tokenizer 和 model
    tokenizer, base_model = load_tokenizer_and_model(config)

    # Step 2. 注入 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.QUESTION_ANS,
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
    train_dataset, val_dataset = load_datasets(config, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])

    # Step 4. 进行训练
    config['train']['steps_per_epoch'] = len(train_loader)
    trainer = QuestionAnsweringTrainer(model, config, device, tokenizer)
    trainer.train(train_loader, val_loader)

    print(f"\n🚀 TensorBoard started! Run this command to view logs:\n")
    print(f"   tensorboard --logdir={config['output']['log_dir']}")
    print("Then open http://localhost:6006 in your browser.\n")

if __name__ == "__main__":
    try:
        import sys
        # 👇 !python train_lora/train_lora_squad.py --config configs/single_lora_squad.yaml

        # 仅当在 Colab 或 Jupyter 环境下运行时 mock sys.argv
        if 'google.colab' in sys.modules:
            sys.argv = ['train_lora_squad.py', '--config', '/content/dynamo/configs/single_lora_squad.yaml']
        
        args = parse_args()
        main(args.config)
    except Exception as e:
        print(f"[FATAL] Uncaught exception: {e}")
        import traceback
        traceback.print_exc()
