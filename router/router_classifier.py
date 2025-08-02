# router/router_classifer.py
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os
import torch
import json
import torch.nn.functional as F

class RouterClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_task=7, temperature=1.0, backbone_name="roberta-base"):
        super().__init__()
        self.temperature = temperature
        self.backbone_name= backbone_name
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.hidden_size = hidden_size
        self.num_task = num_task

        self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(hidden_size, num_task))

    def forward(self, input_ids,  attention_mask=None, return_logits=False):
        # 允许 attention_mask 为 None（兼容性更强）
        if attention_mask is None:
            attention_mask = (input_ids != self.backbone.config.pad_token_id).long()

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
        # logits = self.classifier(cls_token)

        #尝试用 Mean Pooling 替换掉当前的 [CLS] token 作为 Router 分类输入
        # ✅ Mean Pooling 替代 CLS
        last_hidden = outputs.last_hidden_state  # shape: [B, T, H]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()  # shape: [B, T, H]
        summed = torch.sum(last_hidden * mask, dim=1)  # sum over tokens
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)  # avoid division by zero
        mean_pooled = summed / counts  # shape: [B, H]

        logits = self.classifier(mean_pooled)  # use mean pooled embeddings

        logits = logits / self.temperature
        if return_logits:
            return logits
        return F.softmax(logits, dim=-1)
    
    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        # 如果未来你需要部署推理服务
        tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)
        tokenizer.save_pretrained(save_dir)

        # 保存 backbone
        self.backbone.save_pretrained(save_dir)

        # 保存 classifier head
        torch.save(self.classifier.state_dict(), os.path.join(save_dir, "classifier_head.pth"))
        
        # 保存 config（自定义字段）
        config = {
            "backbone_name": self.backbone.config._name_or_path,
            "temperature": self.temperature,
            "hidden_size": self.hidden_size,
            "num_task": self.num_task
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
