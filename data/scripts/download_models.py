import os
from transformers import AutoModel
from huggingface_hub import hf_hub_download

# 你的模型配置清单（你也可以读入CSV或YAML）
MODEL_MAPPING = {
    "adapter_agnews": "distilbert-base-uncased",
    "adapter_sst2": "cardiffnlp/twitter-roberta-base-sentiment",
    "adapter_mnli": "roberta-large-mnli",
    "adapter_qqp": "bert-base-uncased",
    "adapter_conll03": "dslim/bert-base-NER",
    "adapter_squad": "deepset/roberta-large-squad2",
    "adapter_xsum": "google/pegasus-xsum"
}

# 基础路径（你可以改成绝对路径）
BASE_DIR = r"C:\Users\xiuxiuli.SSNC-CORP\Desktop\learn\567ML\dynamo\DynamoRouterCheckpoints"

def download_and_merge():
    for adapter_dir, base_model_name in MODEL_MAPPING.items():
        target_dir = os.path.join(BASE_DIR, adapter_dir)
        model_file = os.path.join(target_dir, "pytorch_model.bin")
        safetensor_file = os.path.join(target_dir, "model.safetensors")

        if os.path.exists(model_file) or os.path.exists(safetensor_file):
            print(f"[✅] {adapter_dir} already contains base model, skipping...")
            continue

        print(f"[⬇️ ] Downloading base model for {adapter_dir} → {base_model_name}")
        try:
            model = AutoModel.from_pretrained(base_model_name)
            model.save_pretrained(target_dir)
            print(f"[✅] Saved base model to {target_dir}")
        except Exception as e:
            print(f"[❌] Failed to download {base_model_name}: {e}")

if __name__ == "__main__":
    download_and_merge()
