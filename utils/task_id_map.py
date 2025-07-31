# utils/task_id_map.py
import json

def load_task_id_map(path="configs/task_id_map.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_task2id(path="configs/task_id_map.json"):
    return load_task_id_map(path)

# 0 ='sst2'
# 1 ='mnli'
# 2 ='qqp'
# 3 ='squad'
# 4 ='xsum'
# 5 ='agnews'
# 6 ='conll2003'
def get_id2task(path="configs/task_id_map.json"):
    task2id = load_task_id_map(path)
    return {v: k for k, v in task2id.items()}