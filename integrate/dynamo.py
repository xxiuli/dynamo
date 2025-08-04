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
        # 因为 text 是统一拼接好的字符串，直接喂 tokenizer 即可
        if not isinstance(text, str):
            raise ValueError(f"[ERROR] Input text must be str, got {type(text)}")
        return tokenizer(text, return_tensors="pt")
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

    # 假设输入是完整样本 dict（含 text, question, context 等）
    def build_task_inputs(self, sample: dict, task_type: str, tokenizer):
        if task_type == "classification":
            return tokenizer(
                sample["text"], 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

        elif task_type == "qa":
            return tokenizer(
                sample["question"], sample["context"],
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

        elif task_type == "summarization":
            return tokenizer(
                sample["document"],
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

        elif task_type == "ner":
            return tokenizer(
                sample["tokens"],
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=320
            ).to(self.device)

        else:
            raise ValueError(f"[ERROR] Unknown task type: {task_type}")

    def predict(self, sample: dict, top_k: int = 3) -> dict:
        apater_pred_class_name = None

        # Step 1: 用 Router 分发任务
        inputs = self.tokenizer(
            sample['text'], 
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding="max_length"
            )
        input_ids = inputs['input_ids'][:, :512].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # router处理
        logits = self.router(input_ids=input_ids, attention_mask=attention_mask)    # 🛠️ 只传 input_ids 给 Router
        
        probs = F.softmax(logits/ self.temperature, dim=-1)
        pred_task_idx = torch.argmax(probs, dim=1).item()
        pred_task_name = self.id_task_map[pred_task_idx]  # get task name

        # print(f"\n📊 Router 认为这是:{task_name} - {task_idx }任务")
        print(f"\n[🧠] Router 执行-分类-任务-------------------->")
        print("📊 Router logits:", logits.tolist()) # 打印 Router logits 和概率
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

        # 是否Router判断正确
        expected_task_id = sample.get("task_id", None)
        is_correct_router = (pred_task_idx == expected_task_id )

        task_cfg = self.tasks[pred_task_name]
        task_type = task_cfg["task_type"].lower()
        class_names = task_cfg.get("class_names", []) 

        # prepare return info
        result = {
                "text": sample.get("text", ""),
                "task_type": task_type,
                "expected_task_id": sample.get("task_id", None),
                "expected_task_name": sample.get("task_name", None),
                "pred_task_id": pred_task_idx,
                "pred_task": pred_task_name,
                "top_k_router": top_k_results,
                "is_router_correct": is_correct_router
                }

        if is_correct_router:
            # 找出对应的LORA
            adapter_info = self.target_adapters[pred_task_name]
            adapter_tokenizer = adapter_info["tokenizer"]
            adapter_model = adapter_info["model"]

            # Step 2: Adapter 推理
            task_inputs = self.build_task_inputs(sample, task_type, adapter_tokenizer)
            task_inputs = {k: v.to(self.device) for k, v in task_inputs.items()}

            # Step 3: 模型推理 # Step 4: 解码结果
            pred = None
            with torch.no_grad():
                print(f"[🧠] LORA Adapter 执行 {task_type} 任务-------------------->")

                if task_type == "summarization":
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
                
                elif task_type == "qa":
                    outputs = adapter_model(**task_inputs)
                    start_idx = torch.argmax(outputs.start_logits, dim=1)
                    end_idx = torch.argmax(outputs.end_logits, dim=1)
                    pred_tokens = task_inputs["input_ids"][0][start_idx: end_idx + 1]
                    pred = adapter_tokenizer.decode(pred_tokens, skip_special_tokens=True)

                elif task_type == "classification":
                    outputs = adapter_model(**task_inputs)
                    print(f"[DEBUG] logits: {outputs.logits}")
                    pred = torch.argmax(outputs.logits, dim=-1).item()
                    print(f"[DEBUG] argmax(pred): {pred}")

                elif task_type == "ner":
                    outputs = adapter_model(**task_inputs)  # logits: [1, seq_len, num_labels]
                    predicted_ids = torch.argmax(outputs.logits, dim=-1)  # [1, seq_len]
                    tokens = adapter_tokenizer.convert_ids_to_tokens(task_inputs["input_ids"][0])
                    labels = [adapter_model.config.id2label[idx.item()] for idx in predicted_ids[0]]
                    
                    pred = list(zip(tokens, labels))  # token-label pair
                    preview = list(zip(tokens, labels))[:10]
                    print(f"[📤] LORA Adapter输出（前10对 token-label）: {preview}")

                else:
                    pred = "Unsupported task type"

                if class_names and isinstance(pred, int):
                    apater_pred_class_name = class_names[pred]

                adapter_result ={
                    "expected_label": sample.get("label", None),
                    "predicted_label": pred,
                    "class_names": class_names if class_names else [],
                    "adapter_pred_class_name": apater_pred_class_name or "",
                    "adapter_is_correct": (pred == sample.get("label")) if isinstance(pred, int) and isinstance(sample.get("label"), int) else None
                }

                result.update(adapter_result)

        return result



    


    