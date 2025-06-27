import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizerFast
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# 选择设备：GPU优先
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4

# 数据路径：你的Router用的混合任务训练集
TRAIN_PATH = os.path.join(BASE_DIR, "data", "mix", "train_mix_250626_Jun06.jsonl")
VAL_PATH = os.path.join(BASE_DIR, "data", "mix", "validation_mix_250626_Jun06.jsonl")

# 自定义Dataset类：用来加载Router训练数据
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
        # 把 input 字段进行分词编码
        encoding = self.tokenizer(
            item['input'],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        label = item['task_id']  # 标签就是任务ID：0, 1, 2
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label)
        }

# 定义Router网络：输入是CLS向量，输出是对任务类别的logits
class TaskRouterMLP(nn.Module):
    def __init__(self, hidden_size=768, num_tasks=3):
        super(TaskRouterMLP, self).__init__()
        self.fc = nn.Linear(hidden_size, num_tasks)  # 一个简单全连接层full connection，映射到3个任务 

    # 输入 x 是一个 batch 的 CLS 向量, CLaSsifier TOKEN
    def forward(self, x):
        return self.fc(x)

# Router训练过程
def train(model, roberta, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training Router"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # 冻结RoBERTa，只做forward，不算梯度
        with torch.no_grad():
            outputs = roberta(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            cls_emb = outputs.last_hidden_state[:, 0, :]  # 取CLS向量

        logits = model(cls_emb)  # Router前向传播，输出logits
        loss = criterion(logits, batch["labels"])  # 计算loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)

# Router验证过程
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

    # 加载Router数据集
    train_set = TaskRouterDataset(TRAIN_PATH, tokenizer)
    val_set = TaskRouterDataset(VAL_PATH, tokenizer)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # 加载RoBERTa主干模型（不带分类头）
    roberta = RobertaModel.from_pretrained("roberta-base")
    for param in roberta.parameters():
        param.requires_grad = False  # 冻结RoBERTa，不参与训练
    roberta.to(DEVICE)

    # 初始化Router网络
    model = TaskRouterMLP()
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train(model, roberta, train_loader, optimizer, criterion)
        val_loss, acc = evaluation(model, roberta, val_loader, criterion)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")

if __name__ == '__main__':
    main()
