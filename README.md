# 🚀 LoRA Fine-Tuning Platform for NLP Tasks

This project provides a **config-driven, extensible training framework** for fine-tuning HuggingFace models with **LoRA adapters** on multiple NLP tasks, including classification, NER, QA, and summarization.

---

## 🧠 Supported Tasks

| Task Type        | Examples     | Trainer Class               | Dataset Loader                   |
|------------------|--------------|-----------------------------|----------------------------------|
| Single Sentence Classification | SST2, AGNews | `SingleClassificationTrainer` | `dataset_cls_single.py` |
| Sentence Pair Classification  | MNLI, QQP    | `SingleClassificationTrainer` | `dataset_cls_pair.py`  |
| Named Entity Recognition (NER)| CoNLL-2003   | `TokenClassificationTrainer`  | `dataset_cls_token.py`  |
| Question Answering (QA)       | SQuAD        | `QuestionAnsweringTrainer`    | `dataset_qa_span.py`    |
| Summarization                 | XSum         | `SummarizationTrainer`        | `dataset_summarization.py` |

---

## 📁 Project Structure
dynamo/
├── configs/
│ ├── single_lora_sst2.yaml
│ ├── single_lora_conll03.yaml
│ └── ...
├── data_loaders/
├── trainers/
├── utils/
├── train_lora_cls.py
├── train_lora_conll03.py
├── train_lora_squad.py
└── train_lora_xsum.py


---

## 🔧 How to Run

```bash
# For single sentence classification (e.g., SST2 or AGNews)
python train_lora_cls.py --config configs/single_lora_sst2.yaml

# For pair classification (e.g., MNLI or QQP)
python train_lora_cls.py --config configs/single_lora_mnli.yaml

# For NER
python train_lora_conll03.py --config configs/single_lora_conll03.yaml

# For QA
python train_lora_squad.py --config configs/single_lora_squad.yaml

# For summarization
python train_lora_xsum.py --config configs/single_lora_xsum.yaml

✅ If running in Google Colab, the script will auto-detect and mock sys.argv for convenience.

📦 Sample Config
task_name: sst2
backbone_model: bert-base-uncased

data:
  train_file: ${DATA_ROOT}/sst2_train.json
  val_file: ${DATA_ROOT}/sst2_val.json

train:
  batch_size: 32
  max_seq_length: 128
  num_epochs: 3
  seed: 42
  log_dir: ${DRIVE_ROOT}/runs/sst2

lora:
  r: 8
  alpha: 32
  dropout: 0.1
  target_modules: ["query", "value"]

💡 ${DATA_ROOT} and ${DRIVE_ROOT} will be replaced via apply_path_placeholders() at runtime.

🧰 Features
✅ LoRA fine-tuning via peft with config-based adapter injection

✅ Automatic dataset class routing via task_map.py

✅ Modular trainer classes per task

✅ TensorBoard logging enabled for each run

✅ CLI or programmatic launch supported

✅ Robust error handling and structured config loading

📊 TensorBoard
After training finishes:
tensorboard --logdir=your_log_dir
# Then visit http://localhost:6006

📦 Requirements
pip install torch transformers peft pyyaml tensorboard


✅ TODO (WIP)
 Config-driven data and model loading

 Support multiple tasks with dynamic routing

 Add evaluation and inference pipeline

 Add Optuna or W&B integration for tuning



