import argparse
from utils.setting_utils import load_config
from integrate.dynamo import Dynamo
from utils.setting_utils import apply_path_dynamo
import json
import os

# 写死配置路径和输入输出路径
config_path = "configs/dynamo.yaml"
input_path = "data/end2end_mix/router_test.jsonl"
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
        router_total = 0
        router_correct = 0

        adapter_total = 0
        adapter_correct = 0

        for sample in samples:
            router_total += 1
            adapter_total += 1

            text = sample["text"]
            expected_task_id = sample["task_id"]
            expected_task_name = sample["task_name"]

            # 进行推理
            result = dynamo.predict(sample)

            # 1. ROUTE的判断结果
            predicted_task_id = result["pred_task_id"]
            predicted_task_name = result["pred_task"]

            # 是否Router判断正确
            is_correct_router = result["is_router_correct"]
            if is_correct_router:
                router_correct += 1

            # 查找期望任务在top_k_router中排第几
            top_k_router = result["top_k_router"]
            rank = next((i for i, r in enumerate(top_k_router) if r["task"] == expected_task_name), None)

            print(f"📝 文本: {text[:80]}{'...' if len(text) > 80 else ''}")
            print(f"✅ 期望任务名: {expected_task_name} (id={expected_task_id})")
            print(f"🔍 Router判断这是: {predicted_task_name} (id={predicted_task_id})")
            print(f"🎯 Router是否正确: {'✅' if is_correct_router else '❌'}")
            if rank is not None:
                print(f"📊 正确任务在 Top-K 排名: 第 {rank + 1} 位")
            else:
                print(f"📊 正确任务未出现在 Top-K")

            print(f"🔼 Top-K 路由候选:")
            for i, r in enumerate(top_k_router):
                print(f"🔼 Top-K 路由候选:   {i+1}. {r['task']} (conf: {r['confidence']})")
            
            # 如果ROUTER分类正确，会有ADAPTER部分的RESULT
            if is_correct_router:

                # 2. LORA adapter的结果
                adapter_pred_label = result["predicted_label"]
                expected_label = result["expected_label"]  # 推理集无论是分类任务或SQUA等，都重构了样板结构，都有LABEL
                
                class_names = result["class_names"]
                adapter_pred_class_name = result["adapter_pred_class_name"]

                # 增加容错逻辑（仅分类任务有 class_names）
                if isinstance(expected_label, int) and isinstance(class_names, list) and len(class_names) > expected_label:
                    expected_class = class_names[expected_label]
                else:
                    expected_class = ""

                print(f"✅ 期望 Label: {expected_label} - {expected_class}")
                print(f"🔍 LORA ADAPTER输出Label:  {adapter_pred_label} - {adapter_pred_class_name}")
                
                # 判断 adapter 是否输出正确
                task_type = result["task_type"]

                try:
                    if task_type == "classification" and isinstance(expected_label, int):
                        if result["adapter_is_correct"]:
                            adapter_correct += 1
                            print(f"🎯 ADAPTER是否正确: {'✅'} (预测: {adapter_pred_label} | 正确: {expected_label})")
                        else:
                            print(f"🎯 ADAPTER是否正确: {'❌'} (预测: {adapter_pred_label} | 正确: {expected_label})")
                    else:
                        adapter_total -= 1
                        # 对于有输出文字的类别，暂时无法统计对错。
                        # TODO: 后续用 ROUGE/LCS/EM 替换 QA、NER，summerization 的 adapter 判断

                except Exception as e:
                    print(f"[⚠️] 判断 LoRA 正确性时出错: {e}")
            
            else:
                print("⏭️ Router误判，未执行Adapter预测，跳过Adapter判断")
                print("==================================================\n")
                continue

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print("\n")
            print("==================================================")

        print(f"🧮 Router 准确率: {router_correct}/{router_total} = {router_correct/router_total:.2%}")

        if adapter_total > 0:
            print(f"🧮 LoRA Adapter 准确率: {adapter_correct}/{adapter_total} = {adapter_correct/adapter_total:.2%}")
        else:
            print("🧮 LoRA Adapter 准确率: 无法计算（没有提供标签）")


    print(f"✅ 推理完成，结果保存在 {args.output_json}")


if __name__ == '__main__':
    main()