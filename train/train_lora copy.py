import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from torch.optim import AdamW 
from peft import get_peft_model, LoraConfig, TaskType
#Parameter-Efficient Fine-Tuning
from sklearn.metrics import accuracy_score
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# config
TASK_NAME = "sst2"
TRAIN_PATH = os.path.join(BASE_DIR, "data", "raw", "train_glue_sst2_20250622_192827.json")
VAL_PATH = os.path.join(BASE_DIR, "data", "raw", "validation_glue_sst2_20250622_192827.json")
SAVE_DIR = f"../models/saved_adapters/{TASK_NAME}/"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#dataset
# ✅ 继承 PyTorch 的 Dataset 接口
class SST2Dataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=128):
        # 打开文件，一行一条 json，读取成列表 self.samples
        with open(json_path, "r") as f:
            self.samples = [json.loads(line) for line in f]

        # 保存 tokenizer 和最大长度参数
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # 返回样本总数，供 DataLoader 调用
        return len(self.samples)

    def __getitem__(self, idx):
        # 取出第 idx 个样本
        item = self.samples[idx]

        # 用 tokenizer 编码句子，输出 input_ids 和 attention_mask
        encoding = self.tokenizer(
            item["sentence"],                   # 输入的文本
            truncation=True,                    # 自动截断超过最大长度
            padding="max_length",               # 不足长度自动补全
            max_length=self.max_length,         # 最大长度限制
            return_tensors="pt",                # 返回 PyTorch 张量格式
        )

        # squeeze(0) 是去掉 batch 维度（因为 tokenizer 默认返回的是 [1, seq_len]）
        return {
            "input_ids": encoding["input_ids"].squeeze(0),               # 1D tensor: [seq_len]
            "attention_mask": encoding["attention_mask"].squeeze(0),     # 1D tensor: [seq_len]
            "labels": torch.tensor(item["label"])                        # 标签（0 或 1），转成 tensor
        }
    
def train(model, dataloader, optimizer):
    model.train()   # 设置模型为训练模式，启用 Dropout 等
    total_loss = 0  # 初始化总损失为 0

    # 遍历每一个 mini-batch
    for batch in tqdm(dataloader, desc="Training"):
        # 将 batch 中的所有张量（input_ids、attention_mask、labels）移动到 GPU 或 CPU
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)    # 前向传播，获得模型输出
        loss = outputs.loss         # 提取损失值
        total_loss += loss.item()   # 累加当前 batch 的损失（用于后面计算平均）

        loss.backward()     # 反向传播，计算梯度
        optimizer.step()    # 优化器更新参数（这里只更新 LoRA adapter 的参数）
        optimizer.zero_grad()   # 清除之前的梯度，避免累积

    return total_loss / len(dataloader) # 返回每个 batch 的平均损失

def evaluate(model, dataloader):
    model.eval()    # 设置模型为评估模式，禁用 Dropout 等
    preds, labels = [], []  # 初始化预测值列表和真实标签列表

    # 在不计算梯度的上下文中进行验证，加速推理，节省显存
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 将 batch 中的所有张量移到 DEVICE 上（CPU 或 GPU）
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)    # 前向传播，获得模型输出
            logits = outputs.logits     # 提取 logits（原始未归一化的分类分数）
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())    # 对 logits 做 argmax，得到预测的类别索引
            labels.extend(batch["labels"].cpu().tolist())   # 保存真实标签
    # 计算并返回准确率（preds 和 labels 都是列表）
    return accuracy_score(labels, preds)

def main():
    # 加载 roberta-base 的分词器，用于将文本转换成模型能理解的 input_ids 等张量
    # tokenizer负责切词，转成数字索引
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # 加载训练/验证数据集，并封装成SST2Dataset 类（继承自 PyTorch 的 Dataset），它能被 DataLoader 自动批处理。
    train_dataset = SST2Dataset(TRAIN_PATH, tokenizer)
    val_dataset = SST2Dataset(VAL_PATH, tokenizer)

    # 用 DataLoader 把上面的 Dataset 封装成可以迭代的「mini-batch」，支持并行、打乱、自动装载数据到 GPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 加载带有分类头的 roberta-base 模型，用于情感分类（2 类）。
    base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    
    # 冻结 RoBERTa 主干网络的“参数梯度更新能力”，让它不参与反向传播中的梯度计算和参数更新，只训练后面加上的部分（比如 LoRA adapter 或 classification head）
    # 冻结 RoBERTa 主干网络的参数
    for param in base_model.roberta.parameters():
        param.requires_grad = False
    
    # 配置 LoRA 超参数
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # 指定任务类型是“序列分类”（例如 SST-2 情感分类）
        r=16,                        # LoRA 的秩（rank），低秩矩阵的维度，决定插入参数量的大小
        lora_alpha=32,               # 缩放因子，用于调整 LoRA 输出的影响力，越大越强. Hugging Face 官方建议：lora_alpha 一般设为 r 的 2~4 倍
        lora_dropout=0.1             # 训练时的 Dropout，避免过拟合
    )

    # 使用 LoRA 包装基础模型（例如 RoBERTa），只在少数参数上训练，冻结其余部分
    model = get_peft_model(base_model, lora_config)

    # 把模型放到 GPU 或 CPU（根据 device 自动判断）
    model.to(DEVICE)

    # 使用 AdamW 优化器，只优化 LoRA 插入的参数部分（其他参数被冻结）
    optimizer = AdamW(model.parameters(), lr=LR)

    # 开始训练多个 epoch
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        # 训练一个 epoch，返回平均训练损失
        train_loss = train(model, train_loader, optimizer)

        # 在验证集上评估准确率（evaluate 是你自己写的函数）
        acc = evaluate(model, val_loader)

        # 输出结果
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f}")

    # 保存训练后的 LoRA adapter 权重到本地文件夹
    model.save_pretrained(SAVE_DIR)

    # 同时保存 tokenizer 配置（如 vocab、merges 等），供后续加载使用
    tokenizer.save_pretrained(SAVE_DIR)

    # 提示保存完成
    print(f"✅ Model saved to {SAVE_DIR}")

if __name__ == '__main__':
    main()