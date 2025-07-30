import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from heads.token_classification_head import TokenClassificationHead  # 你需要定义这个 Head
from safetensors.torch import load_file  # ✅ 导入 safetensors loader

class CustomTokenClassificationModel(nn.Module):
    def __init__(self, backbone, num_labels, ignore_mismatched_sizes=False):
        super().__init__()
        self.backbone = backbone
        self.config = self.backbone.config
        hidden_size = self.config.hidden_size
        self.head = TokenClassificationHead(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        sequence_output = outputs.last_hidden_state
        logits = self.head(sequence_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # 保存 backbone
        if hasattr(self.backbone, "base_model") and hasattr(self.backbone.base_model, "model"):
            self.backbone.base_model.model.save_pretrained(save_directory)
            print(f"[✔] PEFT base_model saved to {save_directory}")
        elif hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(save_directory)
            print(f"[✔] Backbone saved to {save_directory}")
        else:
            print("[⚠️] No valid backbone.save_pretrained found.")

        # 保存 head
        torch.save(self.head.state_dict(), os.path.join(save_directory, "head.pth"))

        # 保存 config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(self.config.to_json_string(indent=2))


    @classmethod
    def from_pretrained(cls, paths, num_labels=None):
        print(f"🟢 [CustomTokenModel] Loading from {paths}")

        config_path = paths['config']
        model_path = paths['model']
        adapter_weights_path = paths['adapter_weight']
        head_path = paths.get('head')  # 有些 NER LoRA 保存流程中没有 head.pth

        # Step 1: 加载 config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Missing config.json at {config_path}")
        config = AutoConfig.from_pretrained(config_path)

        # Step 2: 构造 backbone
        backbone = AutoModel.from_config(config)

        # Step 3: 加载主权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Missing model at {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        backbone.load_state_dict(state_dict)
        print("✅ Loaded backbone weights")

        # Step 4: 获取分类数
        config_num_labels = getattr(config, 'num_labels', None)
        effective_num_labels = num_labels if num_labels is not None else (config_num_labels or 2)

        # Step 5: 构建模型
        model = cls(backbone=backbone, num_labels=effective_num_labels)
        model.config = config

        # Step 6: 加载自定义头（如有）
        if head_path and os.path.exists(head_path):
            model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
            print("✅ Head loaded")

        # Step 7: 加载 LoRA Adapter（safetensors）
        if os.path.exists(adapter_weights_path):
            print("🧩 Loading LoRA adapter weights...")
            model.load_state_dict(load_file(adapter_weights_path), strict=False)
        else:
            print("⚠️ No adapter_model.safetensors found")

        return model
