import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from torch.optim import AdamW 
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TASK_NAME = "sst2"
TRAIN_PATH = os.path.join(BASE_DIR, "data", "raw", "train_glue_sst2_20250622_192827.json")
VAL_PATH = os.path.join(BASE_DIR, "data", "raw", "validation_glue_sst2_20250622_192827.json")

GOOGLE_DRIVE_DIR = f"/content/drive/MyDrive/dynamo_checkpoints_{timestamp}"
COLAB_LOCAL_DIR = "/content/checkpoints"
os.makedirs(GOOGLE_DRIVE_DIR, exist_ok=True)
os.makedirs(COLAB_LOCAL_DIR, exist_ok=True)

SAVE_EACH_EPOCH = False
SAVE_ZIP = False

BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK_TYPE = TaskType.SEQ_CLS
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

class SST2Dataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=128):
        with open(json_path, "r") as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"])
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

def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    return accuracy_score(labels, preds)

def main():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = SST2Dataset(TRAIN_PATH, tokenizer)
    val_dataset = SST2Dataset(VAL_PATH, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

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
    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train(model, train_loader, optimizer)
        acc = evaluate(model, val_loader)
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f}")

        if SAVE_EACH_EPOCH:
            cur_checkpoint_path = f"lora_{TASK_NAME}_epoch{epoch+1}.pth"
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

    if SAVE_ZIP:
        os.system("zip -r /content/checkpoints_backup.zip /content/checkpoints")

if __name__ == '__main__':
    main()
