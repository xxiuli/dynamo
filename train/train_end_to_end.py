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

# 设备选择
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径配置
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4

MIXED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "mix", "train_mix_250626_Jun06.jsonl")
MIXED_VAL_PATH = os.path.join(BASE_DIR, "data", "mix", "validation_mix_250626_Jun06.jsonl")
SAVE_DIR = os.path.join(BASE_DIR, "end2end_checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- Dataset ----------------
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
        label = item["label"]  # 多任务分类的真实标签，比如 sentiment label 或 QA label 等
        task_id = item["task_id"]  # 告诉Router这条样本属于哪个任务
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label),
            "task_id": torch.tensor(task_id)
        }

# ---------------- Router MLP ----------------
class TaskRouterMLP(nn.Module):
    def __init__(self, hidden_size=768, num_tasks=3):
        super(TaskRouterMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_tasks)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        logits = self.fc2(self.relu(self.fc1(x)))
        logits = logits / self.temperature.clamp(min=0.05)
        return logits

# ---------------- End-to-End Model ----------------
class EndToEndModel(nn.Module):
    def __init__(self, num_tasks):
        super(EndToEndModel, self).__init__()
        # 冻结Roberta主干
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        for param in self.roberta.parameters():
            param.requires_grad = False

        # 加载多个LoRA adapter
        self.adapters = nn.ModuleList()
        for task_id in range(num_tasks):
            base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)  # 每任务2分类，你可以改
            lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1)
            adapter_model = get_peft_model(base_model, lora_config)
            self.adapters.append(adapter_model)

        # Router
        self.router = TaskRouterMLP(hidden_size=768, num_tasks=num_tasks)

    def forward(self, input_ids, attention_mask, task_id):
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Router 预测 task_id logits
        router_logits = self.router(cls_emb)
        predicted_task = torch.argmax(router_logits, dim=1)

        # 为每条样本选择对应 Adapter
        batch_logits = []
        for i in range(input_ids.size(0)):
            chosen_task = predicted_task[i].item()
            adapter = self.adapters[chosen_task]
            adapter_input = {
                "input_ids": input_ids[i].unsqueeze(0),
                "attention_mask": attention_mask[i].unsqueeze(0),
                "labels": None  # 这里不算 loss，只取 logits
            }
            with torch.no_grad():
                adapter_outputs = adapter(**adapter_input)
            batch_logits.append(adapter_outputs.logits)

        return router_logits, torch.cat(batch_logits, dim=0)

# ---------------- 训练 ----------------
def train(model, dataloader, optimizer, criterion_router, criterion_task):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training E2E"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        router_logits, task_logits = model(batch["input_ids"], batch["attention_mask"], batch["task_id"])

        # Router loss: 让 Router 学会预测正确 task_id
        router_loss = criterion_router(router_logits, batch["task_id"])

        # Task loss: 每任务自身的分类/预测损失
        task_loss = criterion_task(task_logits, batch["labels"])

        # 总loss
        loss = router_loss + task_loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion_router, criterion_task):
    model.eval()
    total_loss = 0
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating E2E"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            router_logits, task_logits = model(batch["input_ids"], batch["attention_mask"], batch["task_id"])

            router_loss = criterion_router(router_logits, batch["task_id"])
            task_loss = criterion_task(task_logits, batch["labels"])
            loss = router_loss + task_loss
            total_loss += loss.item()

            preds.extend(torch.argmax(task_logits, dim=1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    acc = accuracy_score(labels, preds)
    return total_loss / len(dataloader), acc

# ---------------- Main ----------------
def main():
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    with open(MIXED_TRAIN_PATH, 'r') as f:
        all_task_ids = [json.loads(line)["task_id"] for line in f]
    num_tasks = len(set(all_task_ids))

    train_dataset = MixedTaskDataset(MIXED_TRAIN_PATH, tokenizer)
    val_dataset = MixedTaskDataset(MIXED_VAL_PATH, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = EndToEndModel(num_tasks=num_tasks).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    criterion_router = nn.CrossEntropyLoss()
    criterion_task = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train(model, train_loader, optimizer, criterion_router, criterion_task)
        val_loss, acc = evaluate(model, val_loader, criterion_router, criterion_task)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(SAVE_DIR, f"best_end2end_epoch{epoch+1}_acc{acc:.4f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at: {save_path}")

if __name__ == '__main__':
    main()
