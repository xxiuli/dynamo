from transformers import AutoModel
import torch.nn as nn

def list_lora_target_modules(model_name: str):
    print(f"\n🔍 Checking LoRA target modules for: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    
    # 遍历所有子模块
    candidate_modules = set()
    for name, module in model.named_modules():
        # 仅筛选出 Linear 层
        if isinstance(module, nn.Linear):
            short_name = name.split(".")[-1]
            candidate_modules.add(short_name)

    print("🧩 Candidate LoRA target_modules:")
    for mod in sorted(candidate_modules):
        print(f"  - {mod}")

# 示例调用
list_lora_target_modules("bert-base-uncased")
list_lora_target_modules("roberta-large")
list_lora_target_modules("google/pegasus-xsum")
