import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import torch

def predict(model_dir, text):
    # Load tokenizer and PEFT config
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    peft_config = PeftConfig.from_pretrained(model_dir)

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,  # 自动从 adapter_config.json 读取
        num_labels=2,   # For SST-2 
        local_files_only=True
    )

     # Load adapter weights
     # 把你训练好的 LoRA adapter “注入” 到 base model 中，组成最终的推理模型
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    #prepare input
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    #run infernece
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    label_map = {0: "Negative", 1: "Positive"}
    print(f"\n✅ Text: \"{text}\"\n🧠 Prediction: {label_map[predicted_class]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help='Path to saved LoRA model')
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    args = parser.parse_args()

    predict(args.model_dir, args.text)