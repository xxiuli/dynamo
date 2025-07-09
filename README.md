
---

# DynamoRouter: A Multi-Task LoRA Dynamic Routing Transformer System

🚀 **DynamoRouter** is a multi-task learning system built on **RoBERTa + LoRA adapters + dynamic task routing**.
It supports **task-specific fine-tuning** with **parameter-efficient PEFT methods**, and performs **end-to-end dynamic routing across multiple NLP tasks**.

---

## 🧱 Project Architecture Overview

```
[ Input Text ]
      │
[ Frozen RoBERTa Encoder ]
      │
[ Task Router (MLP) ]
      │
[ Selected LoRA Adapter ] → [ Task-Specific Output Head (Classification / QA / Summarization) ]
      │
[ Loss Calculation (Per Task) ]
```

---

## 🎯 Supported Tasks (Current Setup)

| Task  | Dataset                                                             | LoRA TaskType               |
| ----- | ------------------------------------------------------------------- | --------------------------- |
| SST-2 | Sentiment Classification                                            | SEQ\_CLS                    |
| SQuAD | Question Answering (Span Prediction)                                | QUESTION\_ANS               |
| XSum  | Summarization / Binary Classification (depending on implementation) | SEQ\_CLS or SEQ\_2\_SEQ\_LM |

> ✅ Easily extendable for more tasks!

---

## ✅ Features

* ✅ **Multi-Task Learning (MTL)**
* ✅ **Task-Specific LoRA Adapters**
* ✅ **Dynamic Task Router (MLP-based)**
* ✅ **Multi-Head End-to-End Training**
* ✅ **Multi-Loss Support**
* ✅ **Config-Driven Training Pipelines**
* ✅ **Google Colab + Drive Integration**
* ✅ **PEFT (Parameter-Efficient Fine-Tuning)**

---

## 📂 Project Structure

```
DynamoRouter/
├── configs/                  # LoRA and training hyperparameter configs
│     ├── lora_sst2.json
│     ├── lora_squad.json
│     └── lora_xsum.json
├── data/
│     └── raw/                 # Raw datasets (JSONL or JSON format)
├── lora_checkpoints/          # Saved adapter weights
├── router_checkpoints/        # Saved Router MLP weights
├── end2end_checkpoints/       # Full End-to-End training checkpoints
├── train_lora.py              # Single-task LoRA trainer (config-driven)
├── train_router.py            # Router trainer
├── train_end2end.py           # Multi-head End-to-End trainer
└── README.md
```

---

## ⚙️ Training Workflow

| Phase                              | Script             | Description                                      |
| ---------------------------------- | ------------------ | ------------------------------------------------ |
| 1️⃣ Single-task LoRA fine-tuning   | `train_lora.py`    | Fine-tune one adapter per task                   |
| 2️⃣ Router Training                | `train_router.py`  | Train MLP router on mixed task data              |
| 3️⃣ End-to-End Multi-Task Training | `train_end2end.py` | Joint training of router + adapters (multi-head) |

---

## 📌 Example Usage (Colab)

### Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Train SST2 LoRA Adapter:

```bash
python train_lora.py --config_path ./configs/lora_sst2.json
```

### Train Router:

```bash
python train_router.py
```

### End-to-End Training:

```bash
python train_end2end.py
```

---

## 📝 Possible Improvements (Future Work)

* ✅ AdapterFusion / AdapterDrop support
* ✅ Weighted multi-task loss
* ✅ Optuna / W\&B hyperparameter search
* ✅ Model Export for production inference
* ✅ Docker deployment / API serving

---

## 🧑‍💻 Author

**Xiuxiu Li**
*ML Engineer / Full-Stack Developer / MLOps Learner*

---

