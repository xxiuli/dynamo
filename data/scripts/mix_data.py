import json
import random
import os
from datetime import datetime

timestamp = datetime.now().strftime("%y%m%d_%h%m")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SST2_PATH = os.path.join(BASE_DIR,  "end2end_mix", "test_mix_glue_sst2_20250628_113304.json")
SQUAD_PATH = os.path.join(BASE_DIR,  "end2end_mix", "test_mix_squad_20250628_113304.json")
XSUM_PATH = os.path.join(BASE_DIR, "end2end_mix", "test_mix_xsum_20250628_113304.json")

output_path = os.path.join(BASE_DIR,  "end2end_mix", f"test_end2end_{timestamp}.jsonl")

router_samples = []

def load_sst2():
    samples = []
    with open(SST2_PATH, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            samples.append({
                "input": data["sentence"],
                "task_id": 0
            })
    return samples

def load_squad():
    samples = []
    with open(SQUAD_PATH, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            context = data["context"]
            samples.append({
                "input": f"Question: {question} Context: {context}",
                "task_id": 1
            })
    return samples

def load_xsum():
    samples = []
    with open(XSUM_PATH, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            samples.append({
                "input": data["sentence"],
                "task_id": 2
            })
    return samples

sst2 = load_sst2()
squad = load_squad()
xsum = load_xsum()

# 每个任务各取1000条 (你可以根据需要改)
# sampled_sst2 = random.sample(sst2, min(1000, len(sst2)))
# sampled_squad = random.sample(squad, min(1000, len(squad)))
# sampled_xsum = random.sample(xsum, min(1000, len(xsum)))

# router_samples = sampled_sst2 + sampled_squad + sampled_xsum

router_samples = sst2 + squad + xsum
random.shuffle(router_samples)

with open(output_path, 'w') as f:
    for item in router_samples:
        f.write(json.dumps(item) + "\n")

print(f"✅ Router dataset saved at {output_path}, Total samples: {len(router_samples)}")
