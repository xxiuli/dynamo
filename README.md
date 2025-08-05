# 🧠 DynamoRouter: Unified Multi-Task NLP with LoRA and Task Routing

> A self-initiated Machine Learning project to build a production-grade, modular multi-task NLP system with dynamic task routing, LoRA fine-tuning, and end-to-end joint training.
> > Entire system built **from scratch** (no tutorials, no templates), originally inspired by a Machine Learning course and extended with industry-grade engineering practices.

---

## 🚀 Overview

**DynamoRouter** is a scalable multi-task NLP framework that supports:

- ✅ Text Classification
- ✅ Natural Language Inference
- ✅ Question Answering (SQuAD)
- ✅ Summarization (XSum)
- ✅ Named Entity Recognition

It uses:

- A **Router classifier** to predict task ID dynamically  
- **LoRA adapters** for parameter-efficient fine-tuning  
- Modular **Task Heads** for each task  
- Configurable **training/inference pipeline** ready for deployment

---

## 🧠 Key Concepts

- **Parameter-Efficient Fine-Tuning (LoRA)**  
  Each task uses its own LoRA adapter, enabling scalable fine-tuning with minimal GPU memory usage.

- **Dynamic Task Routing**  
  A lightweight MLP Router automatically predicts the task and dispatches input to the correct adapter + head.

- **Modular Training Pipeline**  
  All training phases—single-task, router, and multi-task joint training—are fully modular and config-driven.

- **Data-Centric Foundation**  
  Focused on high-quality input: standardized formats, aggressive filtering, and lightweight augmentation.  
  ↳ *Final performance was achieved through full-stack optimization beyond data, including router tuning, adapter targeting, and task-specific loss strategies.*

---

## 🧱 System Architecture

              +----------------------+
              |   Input Text         |
              +----------------------+
                         ↓
            +--------------------------+
            | MLP Router (predict task)|
            +--------------------------+
                         ↓
          +-----------------------------+
          | Load LoRA Adapter + Head    |
          +-----------------------------+
                         ↓
              +---------------------+
              |  Output Prediction  |
              +---------------------+

- Backbone: BERT / RoBERTa / Pegasus (frozen)
- Adapter: LoRA (1 per task)
- Head: classification / QA / summarization
- Router: MLP-based task classifier
- Loss: dynamically selected based on task

---

## 📚 Supported Tasks & Datasets

| Task                 | Dataset      | Phase 1 (Adapter) | Phase 2 (Router & MTL) | Adapter Backbone         |
|----------------------|--------------|-------------------|-------------------------|--------------------------|
| Sentiment (CLS)      | SST-2        | 5K                | 1K                      | `bert-base-uncased`      |
| Natural Language Inference | MNLI   | 5K                | 1K                      | `roberta-large-mnli`     |
| Paraphrase (QQP)     | QQP          | 5K                | 1K                      | `bert-base-uncased`      |
| Question Answering   | SQuAD v1     | 50K               | 9K                      | `deepset/roberta-squad2` |
| Summarization        | XSum         | 5K                | 1K                      | `google/pegasus-xsum`    |
| News Classification  | AGNews       | 5K                | 1K                      | `bert-base-uncased`      |
| Named Entity Recognition | CoNLL-2003 | Full            | Full                    | `dslim/bert-base-NER`    |

---

## 🔧 Training Workflow

1. **Dataset Sampling**  
   - Samples 5K training and 1K validation examples per task
   - Converts raw data into unified JSONL format with `text`, `task_id`, and `task_name`

2. **Single-Task LoRA Adapter Training**  
   - Freezes the backbone and trains LoRA adapter + task-specific head

3. **Router Classifier Training**  
   - MLP predicts task_id using [CLS] representation from frozen backbone

4. **End-to-End MTL Training**  
   - Router dynamically selects adapter/head
   - Multi-loss optimization across all tasks

5. **Inference Integration**  
   - Input → Router → Adapter + Head → Output

---

## 📊 Experimental Highlights

- **Router Accuracy**: ~89.2% on validation set
- **Adapter Memory Footprint**: < 1% of full model per task
- **Total Tasks Supported**: 7
- **Ablation Studies**:
  - LoRA vs Full Finetune
  - Static vs Dynamic Routing
  - Router Off Ablation

📈 Tracked with TensorBoard + Weights & Biases  
📂 Configured via `config.yaml` and Hydra

---

## 📦 Repository Structure
.
├── config/ # All YAML configs
├── data/ # Processed datasets
│ ├── scripts/ # sampling scripts
├── src/
│ ├── adapters/ # LoRA adapter loaders
│ ├── heads/ # Task-specific heads
│ ├── routers/ # MLP Router
│ ├── datasets/ # DataLoader modules
│ ├── trainers/ # LoRA / Router / MTL trainers
│ └── inference/ # FastAPI interface
├── checkpoints/ # Saved models
├── logs/ # wandb / TB logs
├── train_lora_cls.py
├── train_router.py
├── train_mtl.py
└── inference.py

---



