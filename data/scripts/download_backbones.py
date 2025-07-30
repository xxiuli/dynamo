import os
import torch
from transformers import AutoModel

# 每个 adapter 对应的原始 backbone 模型名
adapter_backbones = {
    "adapter_agnews": "distilbert-base-uncased",
    "adapter_sst2": "cardiffnlp/twitter-roberta-base-sentiment",
    "adapter_mnli": "roberta-large-mnli",
    "adapter_qqp": "bert-base-uncased",
    "adapter_conll03": "dslim/bert-base-NER",
    "adapter_squad": "deepset/roberta-large-squad2",
    "adapter_xsum": "google/pegasus-xsum"
}

# 路径：请根据你的本地结构自定义
CHECKPOINT_DIR = r"C:\Users\xiuxiuli.SSNC-CORP\Desktop\learn\567ML\dynamo\DynamoRouterCheckpoints"

def fix_backbone(adapter_dir, backbone_name):
    adapter_path = os.path.join(CHECKPOINT_DIR, adapter_dir)
    model_path = os.path.join(adapter_path, "pytorch_model.bin")

    print(f"\n🔧 Checking {adapter_dir}...")

    # 如果已经存在，跳过
    if os.path.exists(model_path):
        print("✅ pytorch_model.bin already exists. Skipping.")
        return

    # 尝试加载并保存 transformer 主模型权重
    try:
        print(f"⏳ Saving backbone: {backbone_name}")
        model = AutoModel.from_pretrained(backbone_name)
        torch.save(model.state_dict(), model_path)
        print("✅ Saved pytorch_model.bin")
    except Exception as e:
        print(f"❌ Failed to save model for {adapter_dir}: {e}")

def main():
    for adapter_name, backbone in adapter_backbones.items():
        fix_backbone(adapter_name, backbone)

if __name__ == "__main__":
    main()
