from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftConfig, PeftModel
import torch

model_dir = "/content/drive/MyDrive/DynamoRouterCheckpoints/sst2/final"
text = 'I really enjoy this movie!'

# load tokenizer and config
tokenizer = AutoTokenizer.from_pretrained(model_dir)
peft_config = PeftConfig.from_pretrained(model_dir)

# Load base model + adapter
base_model = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels=2
)
model = PeftModel.from_pretrained(base_model, model_dir)
model.eval()

# Run inference
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    predict_class = torch.argmaz(outputs.logits, dim=-1).item()

label_map = {0: "Negative", 1: "Positive"}
print(f"✅ Text: \"{text}\"\n🧠 Prediction: {label_map[predicted_class]}")