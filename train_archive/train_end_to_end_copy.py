import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizerFast, RobertaForSequenceClassification
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# 自动选择设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径配置
BASE_DIR = os.path.abspath("/content/dynamo")
MIXED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "end2end_mix", "train_end2end_250628_Jun06.jsonl")
MIXED_VAL_PATH = os.path.join(BASE_DIR, "data", "end2end_mix", "validation_end2end_250628_Jun06.jsonl")
SAVE_DIR = os.path.join(BASE_DIR, "end2end_checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

# Adapter 权重路径（改成你自己的 LoRA adapter 文件路径）
ADAPTER_WEIGHT_PATHS = [
    os.path.join(BASE_DIR, "dynamo_checkpoints", "lora_adapter_xsum_20250627_1424%.pth"),
    os.path.join(BASE_DIR, "dynamo_checkpoints", "lora_adapter_sst2.pth"),
    os.path.join(BASE_DIR, "dynamo_checkpoints", "lora_adapter_squad_20250627_1216%.pth")
]

# Router 权重路径（改成你自己的 Router checkpoint 路径）
ROUTER_WEIGHT_PATH = os.path.join(BASE_DIR, "router_checkpoints", "best_router_epoch3_acc0.6667.pth")

BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4

# --------- 数据集 ---------
class MixedTaskDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        with open(json_path, "r") as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["input"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"]),
            "task_id": torch.tensor(item["task_id"])
        }

# --------- Router MLP ---------
class TaskRouterMLP(nn.Module):
    def __init__(self, hidden_size=768, num_tasks=3):
        super(TaskRouterMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_tasks)
        self.temperature = nn.Parameter(torch.ones(1))  # 学习温度参数，用来控制 softmax 分布张力

    def forward(self, x):
        logits = self.fc2(self.relu(self.fc1(x)))
        logits = logits / self.temperature.clamp(min=0.05)
        return logits

# --------- End-to-End 模型 ---------
class EndToEndModel(nn.Module):
    def __init__(self, num_tasks, adapter_weight_paths, router_weight_path=None):
        super(EndToEndModel, self).__init__()

        # 冻结 Roberta Backbone
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        for param in self.roberta.parameters():
            param.requires_grad = False

        # 初始化 Adapter 列表，并加载单任务训练好的 LoRA adapter 权重
        self.adapters = nn.ModuleList()
        for task_id in range(num_tasks):
            base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
            lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1)
            adapter_model = get_peft_model(base_model, lora_config)
            adapter_model.load_state_dict(torch.load(adapter_weight_paths[task_id], map_location=DEVICE))
            print(f"✅ Loaded adapter weights for task {task_id} from {adapter_weight_paths[task_id]}")
            self.adapters.append(adapter_model)

        # 初始化 Router
        self.router = TaskRouterMLP(hidden_size=768, num_tasks=num_tasks)

        # 加载 Router 权重
        if router_weight_path and os.path.exists(router_weight_path):
            checkpoint = torch.load(router_weight_path, map_location=DEVICE)
            self.router.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded Router weights from {router_weight_path}")
        else:
            print(f"⚠️ Warning: Router weight file not found. Router will train from scratch.")

    def forward(self, input_ids, attention_mask):
        # Roberta 输出 CLS 向量
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Router 预测任务类别
        router_logits = self.router(cls_emb)
        predicted_task = torch.argmax(router_logits, dim=1)

        # 根据 Router 输出选择 Adapter，逐样本处理
        batch_logits = []
        for i in range(input_ids.size(0)):
            chosen_task = predicted_task[i].item()
            adapter = self.adapters[chosen_task]
            adapter_input = {
                "input_ids": input_ids[i].unsqueeze(0),
                "attention_mask": attention_mask[i].unsqueeze(0),
                "labels": None
            }
            adapter_outputs = adapter(**adapter_input)
            batch_logits.append(adapter_outputs.logits)

        return router_logits, torch.cat(batch_logits, dim=0)

# --------- 训练函数 ---------
def train(model, dataloader, optimizer, criterion_router, criterion_task):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training E2E"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        router_logits, task_logits = model(batch["input_ids"], batch["attention_mask"])

        # Router loss
        router_loss = criterion_router(router_logits, batch["task_id"])
        # Task loss
        task_loss = criterion_task(task_logits, batch["labels"])

        # 总 loss
        loss = router_loss + task_loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)

# --------- 验证函数 ---------
def evaluate(model, dataloader, criterion_router, criterion_task):
    model.eval()
    total_loss = 0
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating E2E"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            router_logits, task_logits = model(batch["input_ids"], batch["attention_mask"])
            router_loss = criterion_router(router_logits, batch["task_id"])
            task_loss = criterion_task(task_logits, batch["labels"])
            loss = router_loss + task_loss
            total_loss += loss.item()

            preds.extend(torch.argmax(task_logits, dim=1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    acc = accuracy_score(labels, preds)
    return total_loss / len(dataloader), acc

# --------- Main 主流程 ---------
def main():
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    with open(MIXED_TRAIN_PATH, 'r') as f:
        all_task_ids = [json.loads(line)["task_id"] for line in f]
    num_tasks = len(set(all_task_ids))

    train_dataset = MixedTaskDataset(MIXED_TRAIN_PATH, tokenizer)
    val_dataset = MixedTaskDataset(MIXED_VAL_PATH, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = EndToEndModel(
        num_tasks=num_tasks,
        adapter_weight_paths=ADAPTER_WEIGHT_PATHS,
        router_weight_path=ROUTER_WEIGHT_PATH
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    criterion_router = nn.CrossEntropyLoss()
    criterion_task = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train(model, train_loader, optimizer, criterion_router, criterion_task)
        val_loss, acc = evaluate(model, val_loader, criterion_router, criterion_task)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(SAVE_DIR, f"best_end2end_epoch{epoch+1}_acc{acc:.4f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best End-to-End model at: {save_path}")

if __name__ == '__main__':
    main()
