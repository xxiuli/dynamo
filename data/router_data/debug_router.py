import json

from collections import Counter
with open("data/router_data/router_train.jsonl", encoding='utf-8') as f:
    print(Counter(json.loads(line)["task_name"] for line in f))