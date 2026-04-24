import os
import shlex
import sys
from datetime import datetime
from typing import Any, Iterable, Sequence


def _stringify(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "-"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items()) if value else "-"
    return str(value)


def _getattr(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _section(config: Any, name: str) -> dict:
    if isinstance(config, dict):
        section = config.get(name, {})
        return section if isinstance(section, dict) else {}

    section = getattr(config, f"{name}_cfg", None)
    if isinstance(section, dict):
        return section

    raw_config = getattr(config, "raw_config", {}) or {}
    section = raw_config.get(name, {})
    return section if isinstance(section, dict) else {}


def _field(config: Any, section_name: str, key: str, default: Any = None) -> Any:
    section = _section(config, section_name)
    if key in section:
        return section[key]
    return _getattr(config, key, default)


def _repo_relative(path: Any) -> str:
    if not path:
        return "-"
    path = str(path)
    try:
        if not os.path.isabs(path):
            return path
        relative = os.path.relpath(path, os.getcwd())
        return path if relative.startswith("..") else relative
    except ValueError:
        return path


def _print_box(title: str, rows: Iterable[tuple[str, Any]], width: int = 96) -> None:
    rows = [(label, _stringify(value)) for label, value in rows if value is not None]
    label_width = max([len(label) for label, _ in rows] + [12])
    print()
    print("=" * width)
    print(f"{title}")
    print("=" * width)
    for label, value in rows:
        print(f"{label:<{label_width}} : {value}")
    print("=" * width)


def command_to_string(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def print_launcher_banner(
    *,
    algo: str,
    config_path: str,
    command: Sequence[str],
    config: Any | None = None,
    script: str | None = None,
    dry_run: bool = False,
) -> None:
    rows: list[tuple[str, Any]] = [
        ("Stage", "launcher"),
        ("Algorithm", algo.upper()),
        ("Config", _repo_relative(config_path)),
    ]
    if config is not None:
        rows.extend(
            [
                ("Base model", _field(config, "model", "base_model")),
                ("Task", _field(config, "rl_task", "task")),
                ("Dataset mode", _field(config, "dataset", "mode")),
                ("Datasets", _field(config, "dataset", "tasks")),
            ]
        )
    rows.extend(
        [
                ("Entry target", _repo_relative(script)),
            ("Python", sys.executable),
            ("Dry run", dry_run),
            ("Command", command_to_string(command)),
        ]
    )
    _print_box("C-MORAL RL Launch Plan", rows)


def print_training_summary(config: Any, *, algo: str, config_path: str | None = None) -> None:
    dataset_cfg = _section(config, "dataset")
    rl_task_cfg = _section(config, "rl_task")
    training_batch = int(_field(config, "training", "num_envs", 0) or 0) * int(
        _field(config, "training", "num_return_sequences", 0) or 0
    )
    num_mini_batch = int(_field(config, "training", "num_mini_batch", 0) or 0)
    mini_batch = training_batch // num_mini_batch if num_mini_batch else training_batch

    rows = [
        ("Stage", "training"),
        ("Started at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Algorithm", algo.upper()),
        ("Config", _repo_relative(config_path)),
        ("Base model", _field(config, "model", "base_model")),
        ("Model path", _repo_relative(_field(config, "model", "base_model_path"))),
        ("LoRA path", _repo_relative(_field(config, "model", "lora_adapter_path"))),
        ("Use LoRA", _field(config, "model", "use_lora")),
        ("Device", _field(config, "model", "device")),
        ("Task", rl_task_cfg.get("task", _getattr(config, "task", "-"))),
        ("Dataset mode", dataset_cfg.get("mode", "full")),
        ("Datasets", dataset_cfg.get("tasks", "all")),
        ("Reward mode", rl_task_cfg.get("reward_mode", "absolute")),
        ("Aggregation", rl_task_cfg.get("reward_aggregation", "mean")),
        ("Similarity", rl_task_cfg.get("use_similarity", True)),
        ("Props delta", rl_task_cfg.get("use_props_delta", True)),
        ("Props range", rl_task_cfg.get("use_props_range", True)),
        ("Iterations", _field(config, "training", "num_iterations")),
        ("PPO epochs", _field(config, "training", "ppo_epochs")),
        ("Train batch", training_batch),
        ("Mini batch", mini_batch),
        ("Actor lr", _field(config, "training", "lr_actor")),
        ("Init KL coef", _field(config, "training", "init_kl_coef")),
        ("Eval interval", _field(config, "logs", "eval_interval")),
        ("Save interval", _field(config, "logs", "save_interval")),
        ("Logging backend", _field(config, "logs", "logging_backend")),
        ("Runs dir", _repo_relative(_field(config, "logs", "runs_dir"))),
        ("Logs dir", _repo_relative(_field(config, "logs", "logs_dir"))),
    ]
    _print_box("C-MORAL Training Configuration", rows)


def log_step(name: str, message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{name}] {message}")
