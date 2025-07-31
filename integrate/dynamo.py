from model.custom_token_model import CustomTokenClassificationModel
from utils.setting_utils import load_config
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForSeq2SeqLM
from router.router_classifier import RouterClassifier
from model.custom_cls_model import CustomClassificationModel
from peft import PeftModel
import torch
import torch.nn.functional as F
from utils.setting_utils import apply_path_placeholders
import os
from utils.task_id_map import get_id2task, get_task2id

def load_router(router_cfg):
    # 使用 router.tokenizer 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(router_cfg['tokenizer'])

    # 初始化 Router 模型结构
    model = RouterClassifier.from_pretrained(router_cfg['checkpoint_path'])

    return tokenizer, model

def get_all_adapters(tasks, device):
    target_adapters = {}
    tokenizer_cache = {}
    
    for task_name, task_cfg in tasks.items():
        adapter_cfg = load_config(task_cfg['config_path'])
        adapter_cfg = apply_path_placeholders(adapter_cfg)

        trained_lora_dir = task_cfg['adapter_path']

        # tokenizer 缓存复用
        if trained_lora_dir in tokenizer_cache:
            tokenizer = tokenizer_cache[trained_lora_dir]
        else:
            tokenizer = AutoTokenizer.from_pretrained(trained_lora_dir)
            tokenizer_cache[trained_lora_dir] = tokenizer

        model = load_adapter_model(task_cfg, adapter_cfg, device)

        target_adapters[task_name] = {
                "tokenizer": tokenizer,
                "model": model
            }

    return target_adapters

def load_adapter_model(task_cfg, adapter_cfg, device):
    task_type = task_cfg['task_type'].lower()

    if task_type == 'classification':
        # ${DRIVE_ROOT}/DynamoRouterCheckpoints/adapter_agnews
        model = CustomClassificationModel.from_pretrained(
            task_cfg['model_paths'],
            num_labels=adapter_cfg['num_labels']
            ).to(device)
        return model
    
    elif task_type == 'ner':
        model = CustomTokenClassificationModel.from_pretrained(
            task_cfg['model_paths'],
            num_labels=adapter_cfg['num_labels']
        ).to(device)
        return model

    elif task_type == 'qa':
        # 问答任务：使用 AutoModelForQuestionAnswering
        base_model = AutoModelForQuestionAnswering.from_pretrained(
            task_cfg['adapter_path']
        ).to(device)

        model = PeftModel.from_pretrained(base_model, task_cfg['adapter_path']).to(device)
        model.eval()
        return model

    elif task_type == 'summarization':
        # 摘要任务：使用 Seq2Seq 模型架构
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            task_cfg['adapter_path']
        ).to(device)

        model = PeftModel.from_pretrained(base_model, task_cfg['adapter_path']).to(device)
        model.eval()
        return model
    else:
        raise ValueError(f"[ERROR] Unknown task type: {task_type}")
    
def preprocess_data(text: str, task_type: str, tokenizer):
    if task_type == "classification":
        return tokenizer(text, return_tensors="pt")
    elif task_type == "qa":
        # 暂用模板（后续建议由真实问答结构代替）
        return tokenizer({"question": "What is X?", "context": text}, return_tensors="pt")
    elif task_type == "summarization":
        return tokenizer(text, return_tensors="pt")
    else:
        raise ValueError(f"[ERROR] Unknown task type: {task_type}")
    
def build_path(task_name, task_cfg):
    adapter_dir = task_cfg['adapter_path']
    paths = {
        "config": os.path.join(adapter_dir, "config.json"),
        "model": os.path.join(adapter_dir, "pytorch_model.bin"),
        "head": os.path.join(adapter_dir, "head.pth"),
        "adapter_weight": os.path.join(adapter_dir, f"adapter_{task_name}.safetensors"),
    }
    return paths

class Dynamo:
    def __init__(self, config: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.router = load_router(config["router"])
        self.tasks = config["tasks"]
        self.target_adapters = get_all_adapters(self.tasks, self.device)
        self.task_id_map = get_task2id()
        self.id_task_map = get_id2task()
        self.temperature = config['router']['temperature']

    def predict(self, text: str, top_k: int = 3) -> dict:
        # Step 1: 用 Router 分发任务
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        logits = self.router(input_ids=input_ids, attention_mask=attention_mask)    # 🛠️ 只传 input_ids 给 Router
        probs = F.softmax(logits/ self.temperature, dim=-1)

        task_idx = torch.argmax(probs, dim=1).item()

        # 找到任务名（保持顺序一致）
        task_name = self.id_task_map[task_idx]

        print(f"\n📊 Router 认为这是:{task_name} - {task_idx }任务")
        task_cfg = self.tasks[task_name]

        adapter_info = self.target_adapters[task_name]
        adapter_tokenizer = adapter_info["tokenizer"]
        adapter_model = adapter_info["model"]

        # 打印 Router logits 和概率
        print("\n📊 Router logits:", logits.tolist())
        print("📈 Router softmax:", probs.tolist()) 

         # 输出 Top-k 路由候选
        top_probs, top_indices = probs[0].topk(k=min(top_k, len(self.tasks)))
        top_k_results = [
            {
                "task": self.id_task_map[i.item()],
                "confidence": round(top_probs[j].item(), 4)
            }
            for j, i in enumerate(top_indices)
        ]

        # Step 2: Adapter 推理
        task_type = task_cfg["task_type"].lower()
        task_inputs = preprocess_data(text, task_type, adapter_tokenizer)
        task_inputs = {k: v.to(self.device) for k, v in task_inputs.items()}
        print(f"[🧪] 输入模型的字段: {list(task_inputs.keys())}")


        # Step 3: 模型推理 # Step 4: 解码结果
        with torch.no_grad():
            if task_type == "summarization":
                print("[🧠 LORA Adapter] 使用 generate 进行 summarization 推理")
                for key in ["decoder_input_ids", "decoder_inputs_embeds"]:
                    if key in task_inputs:
                        print(f"[⚠️] 移除冲突字段：{key}")
                        del task_inputs[key]
                pred_ids = adapter_model.generate(
                    input_ids=task_inputs["input_ids"],
                    attention_mask=task_inputs.get("attention_mask"),
                    max_length=60,
                    num_beams=4,
                    early_stopping=True
                )
                pred = adapter_tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
                print(f"📤 Adapter输出（摘要）: {pred}")
            
            elif task_type == "qa":
                print("[🧠 LORA Adapter] 执行问答任务（extractive QA）")
                outputs = adapter_model(**task_inputs)
                start_idx = torch.argmax(outputs.start_logits, dim=1)
                end_idx = torch.argmax(outputs.end_logits, dim=1)
                pred_tokens = task_inputs["input_ids"][0][start_idx: end_idx + 1]
                pred = adapter_tokenizer.decode(pred_tokens, skip_special_tokens=True)
                print(f"📤 Adapter输出（类别索引）: {pred}")

            elif task_type == "classification":
                print("[🧠 LORA Adapter] 执行classification分类任务")
                outputs = adapter_model(**task_inputs)
                pred = torch.argmax(outputs.logits, dim=-1).item()
                print(f"📤 Adapter输出（类别索引）: {pred}")

            elif task_type == "ner":
                print("[🧠 LORA Adapter] 执行ner命名实体识别任务")
                outputs = adapter_model(**task_inputs)  # logits: [1, seq_len, num_labels]
                predicted_ids = torch.argmax(outputs, dim=-1)  # [1, seq_len]
                tokens = adapter_tokenizer.convert_ids_to_tokens(task_inputs["input_ids"][0])
                labels = [adapter_model.config.id2label[idx.item()] for idx in predicted_ids[0]]
                
                pred = list(zip(tokens, labels))  # token-label pair
                preview = list(zip(tokens, labels))[:10]
                print(f"📤 LORA Adapter输出（前10对 token-label）: {preview}")

            else:
                pred = "Unsupported task type"

        return {
            "text": text,
            "task": task_name,
            "task_id": task_idx,
            "predicted_label": pred,
            "top_k_router": top_k_results
        }


    


    