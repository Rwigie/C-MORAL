import os
from types import SimpleNamespace
from typing import Dict, Any, List, Set

try:
    from omegaconf import OmegaConf
except ImportError as exc:
    raise ImportError("OmegaConf is required for configuration loading. Install via `pip install omegaconf`.") from exc


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_KEYS = {
    "base_model_path",
    "lora_adapter_path",
    "best_adapter_dir",
    "config",
    "task_data_path",
    "full_data_path",
    "logs_dir",
    "runs_dir",
    "checkpoints_dir",
}


def load_config(path: str = None):
    """
    Load the experiment config via OmegaConf and expose commonly used fields as attributes.
    """
    default_path = os.path.join(REPO_ROOT, "configs", "train_config.yaml")
    if path is None and os.path.exists(default_path):
        path = default_path
    if path is None:
        raise FileNotFoundError("No configuration file provided and default configs/train_config.yaml not found.")

    data = _load_config_dict(path)
    if not isinstance(data, dict):
        raise ValueError("Configuration file must define a mapping/dictionary.")

    attrs: Dict[str, Any] = {}

    def merge_section(name: str):
        section = data.get(name)
        if isinstance(section, dict):
            for key, value in section.items():
                attrs[key] = value

    flattened_sections = ["model", "training", "ppo", "generation", "eval", "logs", "wandb"]
    for section in flattened_sections:
        merge_section(section)

    for cfg_name in ["dataset", "rl_task", "reward", "train_env", "eval_env"]:
        section = data.get(cfg_name)
        attrs[f"{cfg_name}_cfg"] = section if isinstance(section, dict) else {}

    if "task" not in attrs:
        rl_task = attrs.get("rl_task_cfg", {})
        if isinstance(rl_task, dict) and "task" in rl_task:
            attrs["task"] = rl_task["task"]

    defaults = {
        "logging_backend": "wandb",
        "save_full_critic": True,
        "use_wandb": True,
        "wandb_config": {},
        "use_lr_scheduler": False,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "min_lr_actor": 1.0e-7,
        "min_lr_critic": 1.0e-7,
    }
    for key, value in defaults.items():
        attrs.setdefault(key, value)

    attrs["raw_config"] = data
    return SimpleNamespace(**attrs)


def _deep_merge(base_obj: Any, override_obj: Any) -> Any:
    if not isinstance(base_obj, dict) or not isinstance(override_obj, dict):
        return override_obj
    out = dict(base_obj)
    for key, value in override_obj.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_include_paths(path_value: Any) -> List[str]:
    if path_value is None:
        return []
    if isinstance(path_value, str):
        return [path_value]
    if isinstance(path_value, list):
        return [p for p in path_value if isinstance(p, str)]
    raise ValueError("`include`/`includes` must be a string or list of strings.")


def _normalize_repo_relative_path(path_value: str) -> str:
    if not path_value or os.path.isabs(path_value):
        return path_value

    normalized = os.path.normpath(path_value)
    parts = normalized.split(os.sep)
    if parts and parts[0] == "Molo":
        normalized = os.path.join(*parts[1:]) if len(parts) > 1 else "."

    return os.path.normpath(os.path.join(REPO_ROOT, normalized))


def _normalize_config_paths(obj: Any, parent_key: str | None = None) -> Any:
    if isinstance(obj, dict):
        return {key: _normalize_config_paths(value, key) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_normalize_config_paths(value, parent_key) for value in obj]
    if isinstance(obj, str) and parent_key in PATH_KEYS:
        return _normalize_repo_relative_path(obj)
    return obj


def _load_config_dict(path: str, visited: Set[str] = None) -> Dict[str, Any]:
    abs_path = os.path.abspath(path)
    if visited is None:
        visited = set()
    if abs_path in visited:
        raise ValueError(f"Cyclic config include detected: {abs_path}")
    visited.add(abs_path)

    raw_conf = OmegaConf.load(abs_path)
    data = OmegaConf.to_container(raw_conf, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must define a mapping/dictionary: {abs_path}")

    include_items = _resolve_include_paths(data.pop("include", None))
    include_items += _resolve_include_paths(data.pop("includes", None))

    merged: Dict[str, Any] = {}
    curr_dir = os.path.dirname(abs_path)
    for rel_path in include_items:
        include_path = rel_path if os.path.isabs(rel_path) else os.path.join(curr_dir, rel_path)
        include_data = _load_config_dict(include_path, visited)
        merged = _deep_merge(merged, include_data)

    merged = _deep_merge(merged, data)
    merged = _normalize_config_paths(merged)
    visited.remove(abs_path)
    return merged
