import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizerFast
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
SAVE_DIR = os.path.join(BASE_DIR, "router_checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, "data", "mix", "train_mix_250626_Jun06.jsonl")
VAL_PATH = os.path.join(BASE_DIR, "data", "mix", "validation_mix_250626_Jun06.jsonl")


# results saving
# 保存路径：Google Drive 和 本地
GOOGLE_DRIVE_DIR= f"/content/drive/MyDrive/dynamo_checkpoints"
COLAB_LOCAL_DIR = "/content/checkpoints"
os.makedirs(GOOGLE_DRIVE_DIR, exist_ok=True)
os.makedirs(COLAB_LOCAL_DIR, exist_ok=True)

# 是否每一个EPOCH保存一次
SAVE_EACH_EPOCH = False
SAVE_ZIP = False

class TaskRouterDataset(Dataset):
    def __init__(self, datapath, tokenizer, max_len=512):
        with open(datapath, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        encoding = self.tokenizer(
            item['input'],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        label = item['task_id']
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label)
        }

class TaskRouterMLP(nn.Module):
    def __init__(self, hidden_size=768, num_tasks=3, use_temperature=True):
        super(TaskRouterMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_tasks)
        self.use_temperature = use_temperature
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        if self.use_temperature:
            logits = logits / self.temperature.clamp(min=0.05)
        return logits

def train(model, roberta, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training Router"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = roberta(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            cls_emb = outputs.last_hidden_state[:, 0, :]
        logits = model(cls_emb)
        loss = criterion(logits, batch["labels"])
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(dataloader)

def evaluation(model, roberta, dataloader, criterion):
    model.eval()
    preds, labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Router"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = roberta(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            cls_emb = outputs.last_hidden_state[:, 0, :]
            logits = model(cls_emb)
            loss = criterion(logits, batch["labels"])
            total_loss += loss.item()
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    acc = accuracy_score(labels, preds)
    return total_loss / len(dataloader), acc

def main():
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    with open(TRAIN_PATH, 'r') as f:
        all_labels = [json.loads(line)["task_id"] for line in f]
    num_tasks = len(set(all_labels))
    print(f"✅ Task type count: {num_tasks}")

    train_set = TaskRouterDataset(TRAIN_PATH, tokenizer)
    val_set = TaskRouterDataset(VAL_PATH, tokenizer)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    frozen_roberta = RobertaModel.from_pretrained("roberta-base")
    for param in frozen_roberta.parameters():
        param.requires_grad = False
    frozen_roberta.to(DEVICE)

    model = TaskRouterMLP(num_tasks=num_tasks)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train(model, frozen_roberta, train_loader, optimizer, criterion)
        val_loss, acc = evaluation(model, frozen_roberta, val_loader, criterion)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(GOOGLE_DRIVE_DIR, f"best_router_epoch{epoch+1}_acc{acc:.4f}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, save_path)
            print(f"✅ Save best model to: {save_path}")

if __name__ == '__main__':
    main()
