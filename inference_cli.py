import argparse
from utils.setting_utils import load_config, apply_path_placeholders
from integrate.dynamo import Dynamo

def parse_args():
    parser = argparse.ArgumentParser(description="DynamoRouter Inference CLI")
    parser.add_argument("text", type=str, required=True, help="Input text to classify or analyze")
    parser.add_argument("--config", type=str, default="configs/dynamo.yaml", help="Path to YAML config file")
    return parser.parse_args()

def main():
   args = parse_args()
   dynamo_cfg = load_config(args.config)
   apply_path_placeholders(dynamo_cfg)
   
   # 初始化 Dynamo 推理器
   dynamo = Dynamo(args.config)

   while True:
        text = input("💬 输入文本（或输入 exit 退出）: ")
        if text.lower() == "exit":
            break

        # 执行预测
        result = dynamo.predict(text)

        # 打印结果
        print("\n🧠 Prediction Result:")
        for key, value in result.items():
            print(f"{key}: {value}")

if __name__ == '__main__':
    main()