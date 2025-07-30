import argparse
from utils.setting_utils import load_config
from integrate.dynamo import Dynamo
from utils.setting_utils import apply_path_dynamo
import json
import os

# 写死配置路径和输入输出路径
config_path = "configs/dynamo.yaml"
input_path = "data/end2end_mix/testset.json"
output_path = "results/inference_results.jsonl"

def parse_args():
    parser = argparse.ArgumentParser(description="DynamoRouter Inference CLI")
    parser.add_argument("--config", type=str, default="configs/dynamo.yaml", help="Path to YAML config file")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    dynamo_cfg = load_config(args.config)

    dynamo_cfg = apply_path_dynamo(dynamo_cfg)
   
    # 初始化 Dynamo 推理器
    dynamo = Dynamo(dynamo_cfg)

    with open(args.input_json, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as out_f:
        for sample in samples:
            result = dynamo.predict(sample["text"])
            out_f.write(json.dumps({
                "text": sample["text"],
                "expected_task": sample.get("expected_task"),
                "predicted_task": result.get("predicted_task"),
                "predicted_label": result.get("label")
            }, ensure_ascii=False) + "\n")

    print(f"✅ 推理完成，结果保存在 {args.output_json}")


if __name__ == '__main__':
    main()