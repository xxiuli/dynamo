# router/head_manager.py
import torch.nn as nn

def get_head(loss_type, hidden_dim=768, num_labels=2):
    if loss_type == "cross_entropy":
        return nn.Linear(hidden_dim, num_labels)
    elif loss_type == "token_classification":
        return nn.Linear(hidden_dim, num_labels)
    elif loss_type == "qa_loss":
        return QAHead(hidden_dim)
    elif loss_type == "summarization":
        return SummarizationHead()
    else:
        raise NotImplementedError(f"Unsupported loss_type: {loss_type}")
    

class QAHead(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output):
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

class SummarizationHead(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.qa_outputs = nn.Linear(hidden_size, 2)