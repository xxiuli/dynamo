# utils/task_map.py

from data_loaders.dataset_cls_single import SingleTextClassificationDataset
from data_loaders.dataset_cls_pair import PairTextClassificationDataset
from data_loaders.dataset_cls_token import TokenClassificationDataset
from data_loaders.dataset_qa import QuestionAnsweringDataset
from data_loaders.dataset_summarization import SummarizationDataset

def get_task_info(task_name):
    task_name = task_name.lower()
    
    if task_name in ['sst2', 'agnews']:
        return {
            "task_type": "classification",
            "dataset_class": SingleTextClassificationDataset,
            "extra_args": {}
        }
    elif task_name in ['mnli', 'qqp']:
        return {
            "task_type": "nli",
            "dataset_class": PairTextClassificationDataset,
            "extra_args": {}
        }
    elif task_name == 'conll03':
        return {
            "task_type": "ner",
            "dataset_class": TokenClassificationDataset,
            "extra_args": {"label2id": None}  # 训练时注入
        }
    elif task_name == 'squad':
        return {
            "task_type": "qa",
            "dataset_class": QuestionAnsweringDataset,
            "extra_args": {"doc_stride": 128}
        }
    elif task_name == 'xsum':
        return {
            "task_type": "summarization",
            "dataset_class": SummarizationDataset,
            "extra_args": {}
        }
    else:
        raise ValueError(f"[ERROR] Unsupported task_name: {task_name}")
