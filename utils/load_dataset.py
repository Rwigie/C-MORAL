import os
import re
from datasets import load_dataset as hf_load_dataset, load_from_disk, concatenate_datasets

from ..props.props_info import MOLO_PROPERTIES


def _normalize_alias(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


DATASET_TASK_ALIASES = {
    "erg": "herg",
    "liver": "liv",
    "mutagenicity": "mutag",
    "ampa": "amp",

}


def _build_alias_map():
    alias_map = {}
    for key, info in MOLO_PROPERTIES.items():
        alias_map[_normalize_alias(key)] = key
        alias_map[_normalize_alias(info.full_name)] = key
        for alias in info.aliases:
            alias_map[_normalize_alias(alias)] = key
    for alias, target in DATASET_TASK_ALIASES.items():
        alias_map[_normalize_alias(alias)] = target
    return alias_map


ALIAS_MAP = _build_alias_map()


def normalize_property_token(token: str) -> str:
    normalized = _normalize_alias(token)
    return ALIAS_MAP.get(normalized, normalized)


def parse_task_sections(task_str: str):
    sections = {}
    if not task_str:
        return sections
    segments = [seg.strip() for seg in task_str.split(",") if seg.strip()]
    for seg in segments:
        if ":" in seg:
            prefix, props_str = seg.split(":", 1)
            prefix = prefix.strip().upper()
        else:
            prefix = "C"
            props_str = seg
        tokens = [tok.strip() for tok in props_str.split("+") if tok.strip()]
        normalized = {normalize_property_token(tok) for tok in tokens}
        if prefix not in sections:
            sections[prefix] = set()
        sections[prefix].update(normalized)
    return sections


def parse_task_keys(task_str: str, prefix: str | None = None):
    sections = parse_task_sections(task_str)
    if prefix is None:
        result = set()
        for values in sections.values():
            result.update(values)
        return result
    return sections.get(prefix.upper(), set())


def load_dataset(save_path="data/MumoInstruct_arrow", split="train"):
    path = os.path.join(save_path, split)
    if os.path.exists(path):
        return load_from_disk(path)
    ds = hf_load_dataset("NingLab/C-MuMOInstruct", split=split)
    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(path)
    return ds


def load_dataset_by_task(task, save_path="data/mumo_by_task", split="train", full_save_path="data/MumoInstruct_arrow"):
    path = os.path.join(save_path, split, task)
    if os.path.exists(path):
        return load_from_disk(path)
    ds = load_dataset(save_path=full_save_path, split=split)
    required_keys = parse_task_keys(task)

    def match_required(sample):
        sample_keys = parse_task_keys(sample["task"], prefix="C")
        # allow any superset if all required keys are present, otherwise enforce subset
        if required_keys.issubset(sample_keys):
            return True
        return sample_keys.issubset(required_keys)

    subsets = ds.filter(match_required)
    os.makedirs(path, exist_ok=True)
    subsets.save_to_disk(path)
    return subsets


def build_datasets(config: dict):
    mode = config["mode"]
    tasks = config.get("tasks", [])
    split = config.get("split", "train")
    eval_ratio = config.get("eval_ratio", 0.1)
    task_save_path = config.get("task_data_path", "data/mumo_by_task")
    full_data_path = config.get("full_data_path", "data/MumoInstruct_arrow")

    if mode == "task":
        datasets = [
            load_dataset_by_task(
                task=task,
                save_path=task_save_path,
                split=split,
                full_save_path=full_data_path,
            )
            for task in tasks
        ]
        combined = concatenate_datasets(datasets)
    elif mode == "full":
        combined = load_dataset(save_path=full_data_path, split=split)
    else:
        raise ValueError("mode must be 'task' or 'full'")

    splitted = combined.train_test_split(test_size=eval_ratio, seed=42)
    train_ds = splitted["train"]
    val_ds = splitted["test"]
    return train_ds, len(train_ds), val_ds, len(val_ds)


from datasets import load_dataset, concatenate_datasets

DATASET_TASK_ALIASES = {
    "herg": "erg",       # 输入 herg -> 转为 erg
    "liv": "liver",      # 输入 liv -> 转为 liver (假设数据集用liver)
    "mutag": "mutagenicity", # 输入 mutag -> 转为 mutagenicity
    "amp": "ampa",       # 输入 amp -> 转为 ampa
    # 如果有其他别名，继续在这里添加...
}

def resolve_dataset_task_name(user_task_str: str) -> str:

    if not user_task_str:
        return user_task_str
    parts = user_task_str.lower().split('+')
    resolved_parts = [DATASET_TASK_ALIASES.get(p, p) for p in parts]
    resolved_parts.sort()
    return "+".join(resolved_parts)

def build_test_datasets(config: dict):
    print("Loading C-MuMOInstruct test set...")
    dataset = load_dataset("NingLab/C-MuMOInstruct")
    test_dataset = dataset['test']
    
    mode = config.get("mode", "task")
    raw_tasks = config.get("tasks", [])
    
    if isinstance(raw_tasks, str):
        raw_tasks = [raw_tasks]
    target_tasks = [resolve_dataset_task_name(t) for t in raw_tasks]
    if raw_tasks:
        print(f"Task Mapping: {raw_tasks} -> {target_tasks}")
    target_setting = config.get("instr_setting", "seen")
    print(f"Building test dataset in mode='{mode}' with setting='{target_setting}'...")

    if mode == "task":
        if not target_tasks:
            raise ValueError("Config must provide 'tasks' list when mode is 'task'.")
        filtered_ds = test_dataset.filter(
            lambda example: (
                example['property_comb'] in target_tasks and 
                example['instr_setting'] == target_setting
            )
        )
    
    elif mode == "full":
        filtered_ds = test_dataset.filter(
            lambda example: example['instr_setting'] == target_setting
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    save_root = config.get("task_data_path", "data/test_mumo")
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, mode, target_setting, "+".join(target_tasks) if target_tasks else "all")
    filtered_ds.save_to_disk(save_dir)

    print(f"Test dataset built. Selected {len(filtered_ds)} samples.")
    return filtered_ds, len(filtered_ds)
