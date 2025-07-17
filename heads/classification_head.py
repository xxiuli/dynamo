# heads/classification_head.py

import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)