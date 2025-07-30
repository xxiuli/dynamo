# router/router_classifer.py
import torch.nn as nn
from transformers import AutoModel
import os
import torch
import json

class RouterClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_task=7, temperature=1.0, backbone_name="roberta-base"):
        super().__init__()
        self.temperature = temperature
        self.backbone_name= backbone_name
        self.backbone = AutoModel.from_pretrained(backbone_name)

        self.classifier = nn.Linear(hidden_size, num_task) #分类头

    def forward(self, input_ids,  attention_mask=None):
        # 允许 attention_mask 为 None（兼容性更强）
        if attention_mask is None:
            attention_mask = (input_ids != self.backbone.config.pad_token_id).long()

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
        logits = self.classifier(cls_token)
        return logits / self.temperature
    
    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        # 保存 backbone
        self.backbone.save_pretrained(save_dir)

        # 保存 classifier head
        torch.save(self.classifier.state_dict(), os.path.join(save_dir, "classifier_head.pth"))
        
        # 保存 config（自定义字段）
        config = {
            "backbone_name": self.backbone.config._name_or_path,
            "temperature": self.temperature,
            "hidden_size": self.classifier.in_features,
            "num_task": self.classifier.out_features
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, load_dir):
        with open(os.path.join(load_dir, "config.json"), "r") as f:
            config = json.load(f)

        # 构建空模型
        model = cls(
            hidden_size=config.get("hidden_size", 768),
            num_task=config.get("num_task", 7),
            temperature=config.get("temperature", 1.0),
            backbone_name=config.get("backbone_name", "roberta-base")
        )
        
        # 加载 backbone
        model.backbone = AutoModel.from_pretrained(model.backbone_name)

         # 加载分类头
        head_path = os.path.join(load_dir, "classifier_head.pth")
        model.classifier.load_state_dict(torch.load(head_path, map_location='cpu'))
        
        model.eval()
        return model
