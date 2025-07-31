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
        total = 0
        correct = 0

        for sample in samples:
            total += 1
            text = sample["text"]
            expected_task_id = sample["task_id"]
            expected_task_name = sample["task_name"]

            result = dynamo.predict(text)

            predicted_task_id = result["task_id"]
            predicted_task_name = result["task"]

            # 是否Router判断正确
            is_correct = (predicted_task_name == expected_task_name)
            if is_correct:
                correct += 1

            # 查找期望任务在top_k_router中排第几
            top_k_router = result["top_k_router"]
            rank = next((i for i, r in enumerate(top_k_router) if r["task"] == expected_task_name), None)

            print("--------------------------------------------------")
            print(f"📝 文本: {text[:80]}{'...' if len(text) > 80 else ''}")
            print(f"✅ 期望任务: {expected_task_name} (id={expected_task_id})")
            print(f"🔍 Router判断: {predicted_task_name} (id={predicted_task_id})")
            print(f"🎯 是否正确: {'✅' if is_correct else '❌'}")
            if rank is not None:
                print(f"📊 正确任务在 Top-K 排名: 第 {rank + 1} 位")
            else:
                print(f"📊 正确任务未出现在 Top-K")

            print(f"🔼 Top-K 路由候选:")
            for i, r in enumerate(top_k_router):
                print(f"   {i+1}. {r['task']} (conf: {r['confidence']})")

            # 写入输出文件
            out_f.write(json.dumps({
                "text": text,
                "expected_task_name": expected_task_name,
                "expected_task_id": expected_task_id,
                "predicted_task_name": predicted_task_name,
                "predicted_task_id": predicted_task_id,
                "is_router_correct": is_correct,
                "top_k_router": top_k_router,
                "top_k_rank_of_expected": rank + 1 if rank is not None else None,
                "predicted_label": result["predicted_label"]
            }, ensure_ascii=False) + "\n")

        print("==================================================")
        print(f"🧮 Router 准确率: {correct}/{total} = {correct/total:.2%}")


    print(f"✅ 推理完成，结果保存在 {args.output_json}")


if __name__ == '__main__':
    main()