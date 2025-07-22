# router/router_classifer.py
import torch.nn as nn
from transformers import AutoModel
import os
import torch
import json

class RouterClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_task=7, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.backbone = AutoModel.from_pretrained("roberta-base")
        self.classifier = nn.Linear(hidden_size, num_task) #分类头

    def forward(self, input_ids, attention_mask):
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
            hidden_size=config["hidden_size"],
            num_task=config["num_task"],
            temperature=config["temperature"]
        )
        
        # 加载 backbone
        model.backbone = AutoModel.from_pretrained(config["backbone_name"])

         # 加载分类头
        head_path = os.path.join(load_dir, "classifier_head.pth")
        model.classifier.load_state_dict(torch.load(head_path, map_location='cpu'))

        
        model.eval()
        return model
