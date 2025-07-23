import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from heads.token_classification_head import TokenClassificationHead  # 你需要定义这个 Head

class CustomTokenClassificationModel(nn.Module):
    def __init__(self, backbone_dir, num_labels, ignore_mismatched_sizes=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes
        )
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
        self.backbone.save_pretrained(save_directory)
        torch.save(self.head.state_dict(), os.path.join(save_directory, "head.pth"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(self.config.to_json_string())

    @classmethod
    def from_pretrained(cls, load_directory, num_labels=None):
        config = AutoConfig.from_pretrained(load_directory)
        config_num_labels = getattr(config, "num_labels", None)
        effective_num_labels = num_labels if num_labels is not None else (config_num_labels or 2)

        model = cls(backbone_dir=load_directory, num_labels=effective_num_labels)
        head_path = os.path.join(load_directory, "head.pth")
        model.head.load_state_dict(torch.load(head_path, map_location="cpu"))

        # ✅ 加载 LoRA adapter（可选）
        adapter_weights_path = os.path.join(load_directory, "adapter_model.safetensors")
        if os.path.exists(adapter_weights_path):
            model.load_state_dict(torch.load(adapter_weights_path, map_location="cpu"), strict=False)

        return model
