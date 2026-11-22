import yaml

def load_yaml(file_path:str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(data, file_path:str) -> dict:
    with open(file_path, "w", encoding="utf-8") as f:
        return yaml.safe_dump(data, f, allow_unicode=True)