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
    data_root = os.environ.get("DATA_ROOT", "/content/dynamo/data")
    drive_root = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive")

    # data_root = os.environ.get("DATA_ROOT", os.path.abspath("data"))        # 本地的 ./data
    # drive_root = os.environ.get("DRIVE_ROOT", os.path.abspath("test"))      # 本地的 ./test

    def replace_path(value):
        return (value.replace("${DATA_ROOT}", data_root)
                    .replace("${DRIVE_ROOT}", drive_root)) if isinstance(value, str) else value

    config['data']['train_file'] = replace_path(config['data']['train_file'])
    config['data']['val_file'] = replace_path(config['data']['val_file'])
    config['output']['save_dir'] = replace_path(config['output']['save_dir'])
    config['output']['log_dir'] = replace_path(config['output']['log_dir'])

    if config['data']['label2id_file']:
        config['data']['label2id_file'] = replace_path(config['data']['label2id_file'])
    
    return config