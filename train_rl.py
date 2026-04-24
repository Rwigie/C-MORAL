import argparse
import os
import subprocess
import sys
import tempfile
from copy import deepcopy

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def parse_args():
    parser = argparse.ArgumentParser(description="Unified RL launcher for GRPO/GDPO.")
    parser.add_argument("--algo", required=True, choices=["grpo", "gdpo"], help="Training algorithm.")
    parser.add_argument("--task", type=str, default=None, help="Task name, e.g. elq/bpq.")
    parser.add_argument("--config", type=str, default=None, help="Explicit config path.")
    parser.add_argument("--exp", type=str, default=None, help="Experiment config name under configs/exp, e.g. elq.")
    parser.add_argument(
        "--skip_test_after_train",
        action="store_true",
        help="Pass-through to underlying train script.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print resolved command.")
    return parser.parse_args()


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _entry_script(repo_root: str, algo: str) -> str:
    script_name = "train_grpo.py" if algo == "grpo" else "train_gdpo.py"
    return os.path.join(repo_root, script_name)


def _load_yaml(path: str):
    if yaml is None:
        raise RuntimeError("PyYAML is required for this operation, but yaml is not available.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base_obj, override_obj):
    if not isinstance(base_obj, dict) or not isinstance(override_obj, dict):
        return deepcopy(override_obj)
    out = deepcopy(base_obj)
    for k, v in override_obj.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _write_temp_yaml(content: dict) -> str:
    if yaml is None:
        raise RuntimeError("PyYAML is required for merged config output, but yaml is not available.")
    fd, path = tempfile.mkstemp(prefix="molo_merged_", suffix=".yaml")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False)
    return path


def _resolve_from_exp(repo_root: str, exp_name: str, algo: str) -> str:
    exp_path = os.path.join(repo_root, "configs", "exp", f"{exp_name}.yaml")
    if not os.path.exists(exp_path):
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")
    data = _load_yaml(exp_path)

    # Legacy mode: direct config mapping
    legacy_key = f"{algo}_config"
    if legacy_key in data:
        rel_path = data.get(legacy_key)
        if not rel_path:
            raise ValueError(f"Missing '{legacy_key}' in experiment config: {exp_path}")
        return os.path.join(repo_root, rel_path)

    # New mode: base + overrides merge
    base_key = f"{algo}_base"
    base_rel = data.get(base_key)
    if not base_rel:
        raise ValueError(f"Missing '{base_key}' in experiment config: {exp_path}")
    base_abs = os.path.join(repo_root, base_rel)
    if not os.path.exists(base_abs):
        raise FileNotFoundError(f"Base config not found: {base_abs}")

    base_cfg = _load_yaml(base_abs)
    merged = deepcopy(base_cfg)
    merged = _deep_merge(merged, data.get("common_overrides", {}) or {})
    merged = _deep_merge(merged, data.get(f"{algo}_overrides", {}) or {})
    return _write_temp_yaml(merged)


def resolve_config_path(repo_root: str, algo: str, config: str = None, task: str = None, exp: str = None) -> str:
    if config:
        cfg_path = config
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(repo_root, cfg_path)
        return cfg_path

    if exp:
        return _resolve_from_exp(repo_root, exp, algo)

    if not task:
        raise ValueError("Please provide one of --config / --exp / --task.")

    if algo == "grpo":
        return os.path.join(repo_root, "configs", f"train_config_mistral_{task}.yaml")
    return os.path.join(repo_root, "configs", "gdpo", f"train_config_mistral_{task}.yaml")


def main():
    args = parse_args()
    root = _repo_root()
    config_path = resolve_config_path(
        repo_root=root,
        algo=args.algo,
        config=args.config,
        task=args.task,
        exp=args.exp,
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    entry_script = _entry_script(root, args.algo)
    cmd = [sys.executable, entry_script, "--config", config_path]
    if args.skip_test_after_train:
        cmd.append("--skip_test_after_train")

    print(f"[RL] algo={args.algo}")
    print(f"[RL] script={entry_script}")
    print(f"[RL] config={config_path}")
    print(f"[RL] command={' '.join(cmd)}")

    if args.dry_run:
        return

    try:
        raise SystemExit(subprocess.call(cmd, cwd=root))
    finally:
        if args.exp and os.path.basename(config_path).startswith("molo_merged_") and os.path.exists(config_path):
            try:
                os.remove(config_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
