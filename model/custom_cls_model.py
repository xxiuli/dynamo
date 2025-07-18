from transformers import AutoModel
from heads.classification_head import ClassificationHead
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import os
import torch

class CustomClassificationModel(nn.Module):
    def __init__(self, backbone_name, num_labels, ignore_mismatched_sizes=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            ignore_mismatched_sizes=ignore_mismatched_sizes
            )
        self.config = self.backbone.config #LoRA 的时候，PeftModel 体系会尝试访问 .config.use_return_dict
        
        hidden_size = self.backbone.config.hidden_size #从预训练模型 config 中读取输出维度
        # 要确保 RoBERTa 的 hidden size 和 Head 的输入维度一样, 最关键的接口一致性原则
        self.head = ClassificationHead(hidden_size, num_labels)

    # 模型整体的 forward , 把数据FORWARD向HEAD
    def forward(self, input_ids, attention_mask=None, labels=None, inputs_embeds=None, **kwargs): 
    # def forward(self, input_ids, attention_mask=None,inputs_embeds=None, **kwargs): 

        # Step 1: 输入进入 RoBERTa（或 BERT）
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,  # ✅ 添加这行，转发给 backbone
            **kwargs
            ) # → Roberta

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
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # 保存 backbone（如 RobertaModel）
        self.backbone.save_pretrained(save_directory)
        # 另外保存你自定义的 head
        torch.save(self.head.state_dict(), os.path.join(save_directory, "head.pth"))
        # 保存 config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(self.config.to_json_string())

    @classmethod #告诉 Python 这个方法是类方法，不是实例方法
    def from_pretrained(cls, load_directory):
        # 1. 加载 backbone（包括 config）
        backbone = AutoModel.from_pretrained(load_directory)
        config = backbone.config
        num_labels = config.num_labels  # 从 config 中读

        # 2. 构建模型
        model = cls(backbone_name=load_directory, num_labels=num_labels)

        # 3. 加载自定义的分类头
        head_path = os.path.join(load_directory, "head.pth")
        model.head.load_state_dict(torch.load(head_path, map_location='cpu'))

        return model

