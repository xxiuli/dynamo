import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from peft import get_peft_model, LoraConfig, TaskType
from datetime import datetime
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score

timestamp = datetime.now().strftime("%Y%m%d_%H%M%")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TASK_NAME = 'xsum'
TRAIN_PATH = os.path.join(BASE_DIR, "data", "raw", "train_xsum_20250622_192827.json")
VAL_PATH = os.path.join(BASE_DIR, "data", "raw", "validation_xsum_20250622_192827.json")

GOOGLE_DRIVE_DIR = f"/content/drive/MyDrive/dynamo_checkpoints_{timestamp}"
COLAB_LOCAL_DIR = "/content/checkpoints"
os.makedirs(GOOGLE_DRIVE_DIR, exist_ok=True)
os.makedirs(COLAB_LOCAL_DIR, exist_ok=True)

SAVE_EACH_EPOCH = False
SAVE_ZIP = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-4

TASK_TYPE = TaskType.SEQ_CLS
LORA_R = 24
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

class XSumDataset(Dataset):
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
            item['sentence'],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        label = item['label']
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label)
        }

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(dataloader)

def evaluation(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(batch['labels'].cpu().tolist())
    acc = accuracy_score(labels, preds)
    return acc

def main():
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    train_set = XSumDataset(TRAIN_PATH, tokenizer)
    val_set = XSumDataset(VAL_PATH, tokenizer)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    for param in base_model.roberta.parameters():
        param.requires_grad = False
    lora_config = LoraConfig(
        task_type=TASK_TYPE,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT
    )
    model = get_peft_model(base_model, lora_config)
    model.to(DEVICE)
    print(f"✅ Model is now on device: {next(model.parameters()).device}")
    optimizer = AdamW(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        print()
        train_loss = train(model, train_loader, optimizer)
        acc = evaluation(model, val_loader)
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f}")
        if SAVE_EACH_EPOCH:
            cur_checkpoint_path = f"lora_{TASK_NAME}_{epoch+1}/{EPOCHS}_{timestamp}.pth"
            torch.save(model.state_dict(), cur_checkpoint_path)
            gle_drive_checkpoint = os.path.join(GOOGLE_DRIVE_DIR, f"lora_{TASK_NAME}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), gle_drive_checkpoint)
            colab_local_checkpoint = os.path.join(COLAB_LOCAL_DIR, f"lora_{TASK_NAME}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), colab_local_checkpoint)
    final_dir = os.path.join(GOOGLE_DRIVE_DIR, f"final_model_{timestamp}")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    torch.save(model.state_dict(), os.path.join(final_dir, f"lora_adapter_{TASK_NAME}_{timestamp}.pth"))
    
    print(f"✅ Final model saved to: {final_dir}")
    if SAVE_ZIP:
        print("✅ Zipping local checkpoints for download...")
        os.system("zip -r /content/checkpoints_backup.zip /content/checkpoints")
        print("✅ Zip archive created at /content/checkpoints_backup.zip")

if __name__ == '__main__':
    main()
