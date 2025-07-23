import torch.nn as nn

class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):  # x: [batch_size, seq_len, hidden_size]
        x = self.dropout(x)
        return self.classifier(x)  # 输出: [batch_size, seq_len, num_labels]
