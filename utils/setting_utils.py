# setting_utils.py
import yaml
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    return parser.parse_args()


def load_config(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] Config file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"[ERROR] YAML format error: {e}")
    
def apply_path_placeholders(config):
    #第二个参数是FALLBACK
    # data_root = os.environ.get("DATA_ROOT", "/content/dynamo/data")
    # drive_root = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive")

    data_root = os.environ.get("DATA_ROOT", os.path.abspath("data"))        # 本地的 ./data
    drive_root = os.environ.get("DRIVE_ROOT", os.path.abspath("test"))      # 本地的 ./test

    def replace_path(value):
        if isinstance(value, str):
            replaced = value.replace("${DATA_ROOT}", data_root).replace("${DRIVE_ROOT}", drive_root)
            # ✅ Windows 修复：去除非法路径开头的 '/'（例如 '/C:/Users/...' -> 'C:/Users/...')
            if os.name == "nt" and replaced.startswith("/") and ":" in replaced:
                replaced = replaced[1:]
            return replaced
        return value
    

    config['data']['train_file'] = replace_path(config['data']['train_file'])
    config['data']['val_file'] = replace_path(config['data']['val_file'])
    config['output']['save_dir'] = replace_path(config['output']['save_dir'])
    config['output']['log_dir'] = replace_path(config['output']['log_dir'])

    if 'label2id_file' in config['data'] and config['data']['label2id_file']:
        config['data']['label2id_file'] = replace_path(config['data']['label2id_file'])
    
    return config

def apply_path_dynamo(config):
    #第二个参数是FALLBACK
    # drive_root = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive")
    drive_root = os.environ.get("DRIVE_ROOT", os.path.abspath("DynamoRouterCheckpoints"))

    def replace_path(value):
        if isinstance(value, str):
            replaced = value.replace("${DRIVE_ROOT}", drive_root)
            if os.name == "nt" and replaced.startswith("/") and ":" in replaced:
                replaced = replaced[1:]
            return replaced
        return value

    # 替换 router.checkpoint_path
    config['router']['checkpoint_path'] = replace_path(config['router']['checkpoint_path'])

    # 替换每个 task 的 adapter_path 和 model_paths
    for task_name, task_cfg in config['tasks'].items():
        if 'adapter_path' in task_cfg:
            task_cfg['adapter_path'] = replace_path(task_cfg['adapter_path'])

        if 'model_paths' in task_cfg:
            for key in task_cfg['model_paths']:
                task_cfg['model_paths'][key] = replace_path(task_cfg['model_paths'][key])

    return config