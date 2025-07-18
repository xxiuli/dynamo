# heads/classification_head.py

import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    # HEAD内部的向前传播层
    # 就是把 [CLS] 向量 dropout 后送入 Linear 得到 logits。
    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)  # Linear(hidden_size → num_labels)