import torch

# ----------------------
# 初始化
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')