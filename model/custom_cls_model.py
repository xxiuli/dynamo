from typing import Optional
from transformers import AutoModel,AutoConfig
from heads.classification_head import ClassificationHead
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import os
import torch
from safetensors.torch import load_file  # ✅ 导入 safetensors loader


class CustomClassificationModel(nn.Module):
    # def __init__(self, backbone, num_labels, ignore_mismatched_sizes=False):
    def __init__(
            self, 
            backbone: Optional[nn.Module] = None, 
            backbone_dir: Optional[str] = None,
            num_labels: int = 2, 
            ignore_mismatched_sizes: bool=False
        ):

        super().__init__()

        if backbone is not None and backbone_dir is not None:
            raise ValueError("只能提供 backbone 或 backbone_dir 中的一个，不可同时提供。")
        
        if backbone is not None:
            self.backbone = backbone
        elif backbone_dir is not None:
            self.backbone = AutoModel.from_pretrained(
                backbone_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                local_files_only=True # ✅ 集成时强制使用本地
            )
        else:
            raise ValueError("必须提供 backbone 或 backbone_dir 其中之一。")
      
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
        pooled_output = sequence_output[:, 0, :]  # ✅ 取 [CLS] 向量
        # Step 3: 输入到分类头（Head 会自动调用它的 forward）
        # logits = 模型输出的每个类别的原始分数（未归一化）
        logits = self.head(pooled_output)             # [batch_size, seq_len, num_labels]     

        loss = None

        # Step 4: 如果有标签，计算损失
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100) #需要展平
            # 经SOFTMAX 归一化
            # loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss_fn(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        # ✅ 如果是 PEFT 模型，保存 base_model.model
        if hasattr(self.backbone, "base_model") and hasattr(self.backbone.base_model, "model"):
            self.backbone.base_model.model.save_pretrained(save_directory)
            print(f"[✔] PEFT base_model saved to {save_directory}")
        elif hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(save_directory)
            print(f"[✔] Backbone saved to {save_directory}")
        else:
            print("[⚠️] No valid backbone.save_pretrained found.")

        # 另外保存你自定义的 head
        torch.save(self.head.state_dict(), os.path.join(save_directory, "head.pth"))
        
        # 保存 config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(self.config.to_json_string(indent=2))  # 可读性更好
    
    @classmethod
    def from_pretrained(cls, paths, num_labels=None):
    # def from_pretrained(cls, load_directory, num_labels=None):
        print(f"🟢 [CustomModel] Loading from {paths}")

        config_path = paths['config']
        model_path = paths['model']
        adapter_weights_path = paths['adapter_weight']

        # ✅ Step 1: 加载 transformer config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Missing config.json in {config_path}")
        config = AutoConfig.from_pretrained(config_path)

        # ✅ Step 2: 构造 backbone 模型结构
        backbone = AutoModel.from_config(config)

        # ✅ Step 3: 加载主模型权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Missing pytorch_model.bin in {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        backbone.load_state_dict(state_dict)

        print("✅ Loaded backbone from config + weights")

        # ✅ Step 4: 获取 num_labels
        config_num_labels = getattr(config, 'num_labels', None)
        effective_num_labels = num_labels if num_labels is not None else (config_num_labels or 2)

        # ✅ Step 5: 构建自定义分类模型
        model = cls(backbone=backbone, num_labels=effective_num_labels)

        model.config = config  # 更新 config（否则用的是旧的）

        # ✅ Step 6: 加载 adapter（LoRA）
        if os.path.exists(adapter_weights_path):
            print("🧩 Loading LoRA adapter weights...")
            model.load_state_dict(load_file(adapter_weights_path), strict=False)
        else:
            print("⚠️ adapter_model.safetensors not found, skipping LoRA load")

         # ✅ Step 7: 加载自定义的分类头（Head）
        head_path = paths["head"]
        if head_path and os.path.exists(head_path):
            print("🧠 Loading head weights from:", head_path)
            model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
        else:
            print("⚠️ head.pth not found, using randomly initialized head.")

        return model

    # @classmethod #告诉 Python 这个方法是类方法，不是实例方法
    # def from_pretrained(cls, load_directory, num_labels=None):
    #     # 1. 加载 backbone（包括 config）
    #     print(f'Check dir:  {load_directory}')

    #     backbone = AutoModel.from_pretrained(load_directory, local_files_only=True)
    #     config = backbone.config

    #     # 尝试从 config 中读 num_labels
    #     config_num_labels = getattr(config, 'num_labels', None)

    #     # 使用优先级：外部传入 > config.json > fallback
    #     effective_num_labels = num_labels if num_labels is not None else (config_num_labels or 2)

    #     # 2. 构建模型（用来自定义的分类头维度）
    #     model = cls(backbone_dir=load_directory, num_labels=effective_num_labels)

    #     # 3. 加载自定义 Head
    #     head_path = os.path.join(load_directory, "head.pth")
    #     model.head.load_state_dict(torch.load(head_path, map_location="cpu"))

    #     # ✅ 5. 加载 LoRA adapter 权重（Q/V 层）
    #     adapter_weights_path = os.path.join(load_directory, "adapter_model.safetensors")
    #     if os.path.exists(adapter_weights_path):
    #         model.load_state_dict(torch.load(adapter_weights_path, map_location="cpu"), strict=False)

    #     return model

