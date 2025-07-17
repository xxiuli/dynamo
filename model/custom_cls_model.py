from transformers import AutoModel
from heads.classification_head import ClassificationHead
import torch.nn as nn

class CustomClassificationModel(nn.Module):
    def __init__(self, backbone_name, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size #从预训练模型 config 中读取输出维度
        # 要确保 RoBERTa 的 hidden size 和 Head 的输入维度一样, 最关键的接口一致性原则
        self.head = ClassificationHead(hidden_size, num_labels)

    # 模型整体的 forward , 把数据FORWARD向HEAD
    def forward(self, input_ids, attention_mask=None, labels=None): 
        # Step 1: 输入进入 RoBERTa（或 BERT）
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask) # → Roberta

        # Step 2: 拿到 [CLS] token 向量（第一个位置）
        cls_output = outputs.last_hidden_state[:, 0] 

        # Step 3: 输入到分类头（Head 会自动调用它的 forward）
        # logits = 模型输出的每个类别的原始分数（未归一化）
        logits = self.head(cls_output)      

        loss = None

        # Step 4: 如果有标签，计算损失
        if labels is not None:
            # 经SOFTMAX 归一化
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"logits": logits, "loss": loss}
        else:
            return {"logits": logits}