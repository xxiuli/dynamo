from utils.setting_utils import load_config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from router.router_classifier import RouterClassifier
from model.custom_cls_model import CustomClassificationModel
from peft import PeftModel
import torch
import torch.nn.functional as F

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
    model_dir = task_cfg['adapter_path']

    if task_type == 'classification':
        # ${DRIVE_ROOT}/DynamoRouterCheckpoints/adapter_agnews
        model = CustomClassificationModel.from_pretrained(
            model_dir,
            num_labels=adapter_cfg['num_labels']
            ).to(device)
        return model

    elif task_type in ['qa', 'summarization']:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
        model = PeftModel.from_pretrained(base_model, model_dir).to(device)
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

class Dynamo:
    def __init__(self, config: str):
        self.router, self.tokenizer = load_router(config["router"])
        self.tasks = config["tasks"]
        self.target_adapters = get_all_adapters(self.tasks)

    def predict(self, text: str, top_k: int = 3) -> dict:
        # Step 1: 用 Router 分发任务
        inputs = self.tokenizer(text, return_tensors='pt')
        logits = self.router(**inputs)
        probs = F.softmax(logits, dim=-1)

        task_idx = torch.argmax(probs, dim=1).item()

        # 找到任务名（保持顺序一致）
        task_name = list(self.tasks.keys())[task_idx]
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
                "task": list(self.tasks.keys())[i],
                "confidence": round(top_probs[j].item(), 4)
            }
            for j, i in enumerate(top_indices)
        ]

        # Step 2: Adapter 推理
        adapter_info = self.target_adapters[task_name]
        adapter_tokenizer = adapter_info["tokenizer"]
        adapter_model = adapter_info["model"]

        task_type = task_cfg["task_type"].lower()
        task_inputs = preprocess_data(text, task_type, adapter_tokenizer)
        task_inputs = {k: v.to(self.device) for k, v in task_inputs.items()}

        # Step 3: 模型推理
        with torch.no_grad():
            outputs = adapter_model(**task_inputs)

        # Step 4: 解码结果
        if task_type == "classification":
            if task_type == "summarization":
                pred_ids = adapter_model.generate(**task_inputs)
                pred = adapter_tokenizer.decode(pred_ids[0], skip_special_tokens=True)

            elif task_type == "qa":
                outputs = adapter_model(**task_inputs)
                start_idx = torch.argmax(outputs.start_logits, dim=1)
                end_idx = torch.argmax(outputs.end_logits, dim=1)
                pred_tokens = task_inputs["input_ids"][0][start_idx: end_idx + 1]
                pred = adapter_tokenizer.decode(pred_tokens, skip_special_tokens=True)

            elif task_type == "classification":
                outputs = adapter_model(**task_inputs)
                pred = torch.argmax(outputs.logits, dim=-1).item()

            else:
                pred = "Unsupported task type"


        return {
            "text": text,
            "task": task_name,
            "task_type": task_type,
            "predicted_label": pred,
            "top_k_router": top_k_results
        }


    


    