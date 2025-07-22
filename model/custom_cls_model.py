from transformers import AutoModel
from heads.classification_head import ClassificationHead
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import os
import torch

class CustomClassificationModel(nn.Module):
    def __init__(self, backbone_dir, num_labels, ignore_mismatched_sizes=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_dir,
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
        sequence_output = outputs.last_hidden_state           # [batch_size, seq_len, hidden_size] 

        # Step 3: 输入到分类头（Head 会自动调用它的 forward）
        # logits = 模型输出的每个类别的原始分数（未归一化）
        logits = self.head(sequence_output)             # [batch_size, seq_len, num_labels]     

        loss = None

        # Step 4: 如果有标签，计算损失
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100) #需要展平
            # 经SOFTMAX 归一化
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
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
    def from_pretrained(cls, load_directory, num_labels=None):
        # 1. 加载 backbone（包括 config）

        backbone = AutoModel.from_pretrained(load_directory)
        config = backbone.config

        # 尝试从 config 中读 num_labels
        config_num_labels = getattr(config, 'num_labels', None)

        # 使用优先级：外部传入 > config.json > fallback
        effective_num_labels = num_labels if num_labels is not None else (config_num_labels or 2)

        # 2. 构建模型（用来自定义的分类头维度）
        model = cls(backbone_dir=load_directory, num_labels=effective_num_labels)

        # 3. 加载自定义 Head
        head_path = os.path.join(load_directory, "head.pth")
        model.head.load_state_dict(torch.load(head_path, map_location="cpu"))

        # ✅ 5. 加载 LoRA adapter 权重（Q/V 层）
        adapter_weights_path = os.path.join(load_directory, "adapter_model.safetensors")
        if os.path.exists(adapter_weights_path):
            model.load_state_dict(torch.load(adapter_weights_path, map_location="cpu"), strict=False)

        return model

