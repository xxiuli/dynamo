import yaml
import torch
import json
from transformers import(AutoModel, AutoTokenizer)
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn
from types import SimpleNamespace
from utils.train_utils import set_seed, freeze_base_model, print_trainable_params
import trainer_cls

#tensorboard --logdir=runs, 然后打开浏览器访问：http://localhost:6006

CONFIG_PATH = "configs/single_lora_sst2.yaml"

class TextClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        with open(file_path, 'r') as f:
            try:
                for line in f:
                    item = json.loads(line)
                    self.samples.append({
                        'text': item['sentence'],
                        'label': item['label']
                    })
            except json.JSONDecodeError as e:
                raise ValueError(f"[ERROR] JSON decoding failed in {file_path}: {e}")

        if not self.samples:
            raise ValueError(f"[ERROR] Dataset is empty: {file_path}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item['text'], 
            truncation=True,
            padding='max_length', 
            max_length = self.max_seq_len,
            return_tensors = 'pt'
        )
        return{
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }
    
def load_config(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] Config file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"[ERROR] Invalid YAML format: {e}")


def main():
    # 加载配置文件，
    config = load_config(CONFIG_PATH)

    # 设置种子，选择 GPU/CPU
    set_seed(config['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    try: 
        # AutoTokenizer 是把文本（句子）转为模型输入所需的 token 序列. tokenize（分词 + 编码）
        tokenizer = AutoTokenizer.from_pretrained(config['backbone_model']) # roberta-base
        # load base model 1. 加载模型
        base_model = AutoModel.from_pretrained(config['backbone_model']) # roberta-base
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load backbone model/tokenizer: {e}")
    
    train_set = TextClassificationDataset(config['data']['train_file'], tokenizer, config['train']['max_seq_length'])
    val_set = TextClassificationDataset(config['data']['val_file'], tokenizer, config['train']['max_seq_length'])

    # 存入LOADER， total/batchsize, 分开很多个小LOAD | 把数据集按 batch 划分，封装成迭代器供训练时使用
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'])
    
    # inject lora 2. 注入 LoRA
    peft_config = LoraConfig(
        task_type = TaskType.SEQ_CLS,
        r = config['lora']['r'],
        lora_alpha = config['lora']['alpha'],
        lora_dropout = config['lora']['dropout'],
        bias = 'none',
        target_modules=config['lora']['target_modules']
    )
    
    try:
        # 把LORA 插入ROBERTA
        model = get_peft_model(base_model, peft_config)
        model.to(device)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to inject LoRA: {e}")

    # Add classification head 3. 添加分类头
    # hidden_size = base_model.config.hidden_size
    # num_labels = config['num_labels']

    # 初始化’LORA+ROBERTA‘合体
    # model = LoRAModelWithClassifier(lora_base, hidden_size, num_labels)

    # 4. 冻结主干（一定要在 optimizer 之前）
    freeze_base_model(model)

    # 5. 打印可训练参数数目（可选）
    print_trainable_params(model)

    # 模型和数据都要移到设备上
    # model.to(device)

    # 自动记录 steps_per_epoch
    config['train']['steps_per_epoch'] = len(train_loader)

    #开始训练
    trainer_instance = trainer_cls.SingleTaskTrainer(model, config, device)
    try:
        trainer_instance.train(train_loader, val_loader, tokenizer=tokenizer)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise  # 可选，或保存日志再退出
    

    print(f"\n🚀 TensorBoard started! Run this command to view logs:\n")
    print(f"   tensorboard --logdir={config['output']['log_dir']}")
    print("Then open http://localhost:6006 in your browser.\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Uncaught exception in main: {e}")