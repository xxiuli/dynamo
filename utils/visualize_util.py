# visualize.py
# 用于可视化 NER（CoNLL-03）、QA（SQuAD）、摘要（XSum）任务训练结果

import os
import json
import matplotlib.pyplot as plt
import argparse

# python visualize.py --task ner --input outputs/ner_f1.json --output_dir plots/
# python visualize.py --task qa --input outputs/qa_f1.json --output_dir plots/
# python visualize.py --task summarization --input outputs/rouge_l.json --output_dir plots/

def plot_metric_curve(metric_dict, title, ylabel, save_path):
    epochs = list(metric_dict.keys())
    values = list(metric_dict.values())
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, values, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


def plot_ner_entity_f1(f1_dict, save_path):
    plt.figure(figsize=(6, 4))
    plt.bar(f1_dict.keys(), f1_dict.values(), color='skyblue')
    plt.title("CoNLL-03 Entity-wise F1 Score")
    plt.ylabel("F1 Score")
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['ner', 'qa', 'summarization'])
    parser.add_argument('--input', type=str, required=True, help='Path to metric JSON file')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.task == 'ner':
        entity_f1 = load_json(args.input)  # format: {"PER": 91.0, "ORG": 84.2, ...}
        save_path = os.path.join(args.output_dir, "ner_entity_f1.png")
        plot_ner_entity_f1(entity_f1, save_path)

    elif args.task == 'qa':
        f1_data = load_json(args.input)  # format: {"1": 61.3, "2": 66.5, ...}
        f1_data = {int(k): v for k, v in f1_data.items()}
        save_path = os.path.join(args.output_dir, "qa_f1_curve.png")
        plot_metric_curve(f1_data, "SQuAD F1 per Epoch", "F1 Score", save_path)

    elif args.task == 'summarization':
        rouge_data = load_json(args.input)  # format: {"1": 33.1, "2": 36.5, ...}
        rouge_data = {int(k): v for k, v in rouge_data.items()}
        save_path = os.path.join(args.output_dir, "rouge_l_curve.png")
        plot_metric_curve(rouge_data, "XSum ROUGE-L per Epoch", "ROUGE-L", save_path)


if __name__ == '__main__':
    main()
