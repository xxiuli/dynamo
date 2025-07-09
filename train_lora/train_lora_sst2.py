import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import json
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from utils.train_utils import set_seed, freeze_base_model, print_trainable_params
import trainer


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
            max_length=self.max_seq_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

# class LoRAModelWithClassifier(nn.Module):
#     def __init__(self, base_model, hidden_size, num_labels):
#         super().__init__()
#         self.base = base_model
#         self.classifier = nn.Linear(hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask, **kwargs):
#         print(f"[DEBUG] kwargs passed to base: {kwargs}")
#         # 手动移除 kwargs 中的非法参数
#         # assert isinstance(self.base, RobertaModel)
#         outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
#         logits = self.classifier(outputs.last_hidden_state[:, 0, :])
#         return SimpleNamespace(logits=logits)

def load_config(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] Config file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"[ERROR] Invalid YAML format: {e}")

def main():
    config = load_config(CONFIG_PATH)
    set_seed(config['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(config['backbone_model'])
        base_model = AutoModelForSequenceClassification.from_pretrained(config['backbone_model'], num_labels=config['num_labels'])
        # base_model = AutoModel.from_pretrained(config['backbone_model'])
        # print(f"[DEBUG] base_model class: {base_model.__class__}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load backbone model/tokenizer: {e}")
    
    train_set = TextClassificationDataset(config['data']['train_file'], tokenizer, config['train']['max_seq_length'])
    val_set = TextClassificationDataset(config['data']['val_file'], tokenizer, config['train']['max_seq_length'])
    
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'])

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        bias='none',
        target_modules=config['lora']['target_modules']
    )
    
    try:
        # lora_base = get_peft_model(base_model, peft_config)
        model = get_peft_model(base_model, peft_config)
        model.to(device)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to inject LoRA: {e}")

    # hidden_size = base_model.config.hidden_size
    # num_labels = config['num_labels']
    # model = LoRAModelWithClassifier(lora_base, hidden_size, num_labels)

    # freeze_base_model(model)
    print_trainable_params(model)

    config['train']['steps_per_epoch'] = len(train_loader)

    trainer_instance = trainer.SingleTaskTrainer(model, config, device)
    try:
        trainer_instance.train(train_loader, val_loader)
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
        print(f"[FATAL] Uncaught exception in main: {e}")
