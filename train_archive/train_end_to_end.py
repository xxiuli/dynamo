import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import RobertaTokenizerFast, RobertaModel, RobertaForSequenceClassification
from datetime import datetime
from peft import LoraConfig, TaskType, get_peft_model

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# data path
BASE_DIR = os.path.abspath("/content/dynamo")
TRAIN_ROUTER_DATA_PATH = os.path.join(BASE_DIR, 'data', 'end2end_mix','train_end2end_250628_Jun06.jsonl')
VAL_ROUTER_DATA_PATH = os.path.join(BASE_DIR, 'data', 'end2end_mix','tvalidation_end2end_250628_Jun06.jsonl')

# .pth save path
MY_DRIVE = f"/content/drive/MyDrive/dynamo_checkpoints"
os.makedirs(MY_DRIVE, exist_ok=True)

# DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pre_trained .pth path: lora, router
ADAPTER_WEIGHT_PATHS = [
    os.path.join(MY_DRIVE, "lora_adapter_xsum_20250627_1424%.pth"),
    os.path.join(MY_DRIVE, "lora_adapter_sst2.pth"),
    os.path.join(MY_DRIVE, "lora_adapter_squad_20250627_1216%.pth")
]

ROUTER_WEIGHT_PATH = os.path.join(MY_DRIVE, "dynamo_checkpoints", "router", "best_router_epoch2_acc0.67.pth")

# TRAIN LOOP CONFIG
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4

# LORA config
LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)

# --------- 数据集 ---------
class MixTaskDataset(Dataset):
    def __init__(self, datapath, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(datapath, 'r') as f:
            self.samples = [ json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        item = self.samples[index]
        encoding =  self.tokenizer(
            item["input"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensor='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze[0],
            'attention_mask': encoding['attention_maske'].squeeze[0],
            'labels': torch.tensor(item['label']),
            'task_id': torch.tensor(item['task_id'])
        }
    
# --------- Router MLP ---------
class TaskRouterMLP(nn.Module):
    def __init__(self, hidden_size=768, num_tasks=3):
        super(TaskRouterMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)

# --------- End-to-End 模型 ---------
class EndToEndModel(nn.Module):
    def __init__(self, num_tasks, adapter_weight_paths, router_wei_path=None):
        super(EndToEndModel, self).__init__()

        # 冻结 Roberta backbone
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        for param in self.roberta.parameters():
            param.requires_grad = False

        # 初始化 adapter 列表， 并加载已经训练好的单任务LORA adapter权重
        self.adapters = nn.ModuleList
        for task_id in range(num_tasks):
            base_model = RobertaForSequenceClassification.from_pretrained('roberta-base')
            lora_config = LORA_CONFIG

            adapter_model = get_peft_model(base_model, lora_config)
            adapter_model.load_state_dict(torch.load(adapter_weight_paths[task_id], map_location=DEVICE))
            print(f"✅ Loaded adapter weights for task {task_id} from {adapter_weight_paths[task_id]}")
            self.adapters.append(adapter_model)

        # 
        self.router = TaskRouterMLP(hidden_size=768, num_tasks=num_tasks)

def main():
    # ------------data-----------
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    train_router_data = MixTaskDataset(TRAIN_ROUTER_DATA_PATH, tokenizer)
    val_router_data = MixTaskDataset(VAL_ROUTER_DATA_PATH, tokenizer)

    train_loader = DataLoader(train_router_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_router_data, batch_size=BATCH_SIZE)

    with open(TRAIN_ROUTER_DATA_PATH, 'r') as f:
        all_task_ids = [json.loads(line)['task_id'] for line in f]
    num_tasks = len(set(all_task_ids))


    model = EndToEndModel(

    ).to(DEVICE)

    print()

if __name__=='__main__':
    main()