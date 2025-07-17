# router/router_classifer.py
import torch.nn as nn
from transformers import AutoModel

class RouterClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_task=7, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.backbone = AutoModel.from_pretrained("roberta-base")
        self.classifier = nn.Linear(hidden_size, num_task)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
        logits = self.classifier(cls_token)
        return logits / self.temperature
