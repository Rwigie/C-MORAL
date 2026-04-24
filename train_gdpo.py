import argparse
import torch
import copy
import numpy as np
import os
import sys
import types
import builtins
import sklearn.svm._classes as new_svm_classes
import json

from functools import partial

if __package__:
    from .rl import GDPOTrainer, GRPORolloutBuffer, AdaptiveKL
    from .rl.ppo.actor import MoloActor
    from .rl.reward import compute_reward
    from .rl.reward_sigmoid import compute_reward_sigmoid
    from .rl.reward_gdpo import compute_reward_gdpo
    from .utils import load_model_tokenizer, build_datasets, build_test_datasets, load_config, check_lora_weights
    from .rl.env import MoloEnv
    try:
        from .test.test_function import MM_test
    except ImportError:
        MM_test = None
else:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    from rl import GDPOTrainer, GRPORolloutBuffer, AdaptiveKL
    from rl.ppo.actor import MoloActor
    from rl.reward import compute_reward
    from rl.reward_sigmoid import compute_reward_sigmoid
    from rl.reward_gdpo import compute_reward_gdpo
    from utils import load_model_tokenizer, build_datasets, build_test_datasets, load_config, check_lora_weights
    from rl.env import MoloEnv
    try:
        from test.test_function import MM_test
    except ImportError:
        MM_test = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to training config JSON/YAML.")
    parser.add_argument(
        "--skip_test_after_train",
        action="store_true",
        help="Skip running test_molo-style evaluation on the test dataset after training.",
    )
    return parser.parse_args()


def _normalize_model_segment(base_model: str) -> str:
    model = (base_model or "").strip().lower()
    if model in {"mistral", "llama"}:
        return model
    return model or "unknown_model"


def _insert_model_segment(path_value: str, model_segment: str, algo_segment: str) -> str:
    if not path_value:
        return path_value

    normalized = os.path.normpath(path_value)
    parts = normalized.split(os.sep)
    parts_lower = [p.lower() for p in parts]

    if model_segment in parts_lower:
        return normalized

    algo_lower = algo_segment.lower()
    if algo_lower in parts_lower:
        idx = parts_lower.index(algo_lower)
        parts.insert(idx, model_segment)
        return os.path.join(*parts)

    return os.path.join(normalized, model_segment, algo_lower)


def _apply_model_grouped_paths(config, algo: str) -> None:
    model_segment = _normalize_model_segment(getattr(config, "base_model", ""))

    if hasattr(config, "runs_dir") and config.runs_dir:
        config.runs_dir = _insert_model_segment(config.runs_dir, model_segment, algo)
    if hasattr(config, "logs_dir") and config.logs_dir:
        config.logs_dir = _insert_model_segment(config.logs_dir, model_segment, algo)


def configuration_plot(config) -> None:
    dataset_cfg = config.dataset_cfg or {}
    rl_task_cfg = config.rl_task_cfg or {}
    training_batch = config.num_envs * config.num_return_sequences
    mini_batch = training_batch // config.num_mini_batch if config.num_mini_batch else training_batch

    print(
        "\n"
        + "=" * 80
        + "\n"
        + "🌟 LLM4LigOpt Training Configuration Summary\n"
        + "=" * 80
        + "\n"
        + f"Base Model              : {config.base_model}\n"
        + f"Model Path              : {config.base_model_path}\n"
        + f"LoRA Path               : {config.lora_adapter_path}\n"
        + f"Use LoRA                : {config.use_lora}\n"
        + f"Dataset Mode            : {dataset_cfg.get('mode', 'full')}\n"
        + f"Datasets                : {dataset_cfg.get('tasks', 'all')}\n"
        + f"RL Task                 : {rl_task_cfg.get('task', config.task)}\n"
        + f"Algorithm               : {config.algorithm}\n"
        + f"Actor lr                : {config.lr_actor}\n"
        + f"Init Kl coef            : {config.init_kl_coef}\n"
        + f"Reward Mode             : {rl_task_cfg.get('reward_mode', 'absolute')}\n"
        + f"Aggregation             : {rl_task_cfg.get('reward_aggregation', 'mean')}\n"
        + f"Use Similarity          : {rl_task_cfg.get('use_similarity', True)}\n"
        + f"Use Props Delta         : {rl_task_cfg.get('use_props_delta', True)}\n"
        + f"Use Props Range         : {rl_task_cfg.get('use_props_range', True)}\n"
        + f"num_iterations          : {config.num_iterations}\n"
        + f"ppo_epochs              : {config.ppo_epochs}\n"
        + f"batch_size              : {training_batch}\n"
        + f"num_envs                : {config.num_envs}\n"
        + f"num_return_sequences    : {config.num_return_sequences}\n"
        + f"mini_batch_size         : {mini_batch}\n"
        + "=" * 80
    )

def _to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.number):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def run_test_after_training(config, trainer, admet_model):
    if MM_test is None:
        print("⚠️ [Testing] test module is not available. Skipping post-training evaluation.")
        return

    raw_cfg = getattr(config, "raw_config", {}) or {}
    dataset_cfg = dict(config.dataset_cfg or {})
    rl_task_cfg = dict(config.rl_task_cfg or {})
    test_cfg = dict(raw_cfg.get("test", {}) or {})
    model_task = rl_task_cfg.get("task", config.task or "")

    if model_task:
        dataset_cfg["tasks"] = model_task
    if "mode" not in dataset_cfg:
        dataset_cfg["mode"] = "task"

    print("📦 [Dataset] Building test dataset...")
    test_ds, test_len = build_test_datasets(dataset_cfg)
    print(f"✅ [Environment]['test'] Ready with {test_len} samples.")

    test_env = MoloEnv(
        dataset=test_ds,
        task=model_task,
        props_delta=rl_task_cfg.get("props_delta", {}),
        props_range=rl_task_cfg.get("props_range", {}),
        reward_weight={},
        include_delta=False,
        include_weight=False,
        random_sample=False,
        device=config.device,
    )

    best_adapter_path = os.path.join(trainer.run_dirs["best_adapter_dir"], "lora_adapter")
    print(f"📦 [Model] Loading best adapter from: {best_adapter_path}")
    test_model, tokenizer = load_model_tokenizer(
        base_model=config.base_model,
        base_model_path=config.base_model_path,
        lora_adapter_path=best_adapter_path,
        use_lora=config.use_lora,
        device=config.device,
    )
    test_actor = MoloActor(test_model, tokenizer, device=config.device)

    num_return_sequences = int(test_cfg.get("num_return_sequences", getattr(config, "eval_num_return_sequences", 1)))
    num_beams = int(test_cfg.get("num_beams", test_cfg.get("num_beam", getattr(config, "eval_num_beams", 1))))
    max_new_tokens = int(test_cfg.get("max_new_tokens", getattr(config, "max_new_tokens", 100)))
    batch_size = int(test_cfg.get("batch_size", 64))
    do_sample = bool(test_cfg.get("do_sample", False))

    print(f"🏆 [Testing] Starting evaluation on test split: task={model_task} ...")
    test_metric = MM_test(
        actor=test_actor,
        env=test_env,
        num_envs=test_len,
        num_return_sequences=20,
        num_beams=20,
        max_new_tokens=max_new_tokens,
        admet_model=admet_model,
        do_sample=False,
        eval_indices=None,
        batch_size=20,
        tasks=model_task,
    )
    print(f"🏆 [Testing] Results: {test_metric}")

    output_path = os.path.join(trainer.run_dirs["runs_dir"], "test_metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": model_task,
                "num_samples": test_len,
                "num_return_sequences": num_return_sequences,
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
                "batch_size": batch_size,
                "do_sample": do_sample,
                "metrics": _to_jsonable(test_metric),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"💾 [Testing] Metrics saved to: {output_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    _apply_model_grouped_paths(config, algo="gdpo")
    configuration_plot(config)
    reward_cfg = dict(config.reward_cfg or config.rl_task_cfg or {})
    if config.task and "task" not in reward_cfg:
        reward_cfg["task"] = config.task

    # reward_fn = partial(
    #     compute_reward,
    #     reward_mode=reward_cfg.get("reward_mode", "absolute"),
    #     aggregation=reward_cfg.get("reward_aggregation", "weighted_sum"),
    #     reward_weight=reward_cfg.get("reward_weight", {}),
    #     use_similarity=reward_cfg.get("use_similarity", True),
    #     use_props_delta=reward_cfg.get("use_props_delta", True),
    #     props_delta=reward_cfg.get("props_delta", {}),
    #     props_delta_penalty=reward_cfg.get("props_delta_penalty", {}),
    #     use_props_range=reward_cfg.get("use_props_range", True),
    #     props_range=reward_cfg.get("props_range", {}),
    #     props_range_penalty=reward_cfg.get("props_range_penalty", {}),
    #     scale=reward_cfg.get("reward_scale", 4.0),
    #     clip=reward_cfg.get("reward_clip", 1.0),
    #     admet_model=reward_cfg.get("admet_model"),
    #     task=reward_cfg.get("task", config.task or "plogp+qed"),
    # )
    
    sys.modules['sklearn.svm.classes'] = new_svm_classes

    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    safe_types = [
        argparse.Namespace,
        np.ndarray,
        np.dtype,
        np.core.multiarray._reconstruct,
        np.dtype('float64').__class__,  
        set, slice, tuple, list, dict, float, int, str, bytes,
        type, types.SimpleNamespace,
        builtins.object,
    ]

    print("Loading ADMET model...")
    with torch.serialization.safe_globals(safe_types):
        from admet_ai import ADMETModel
        admet_model = ADMETModel(num_workers=0)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    
    reward_fn = partial(
        compute_reward_gdpo,
        reward_mode=reward_cfg.get("reward_mode", "absolute"),
        aggregation=reward_cfg.get("reward_aggregation", "weighted_sum"),
        reward_weight=reward_cfg.get("reward_weight", {}),
        use_similarity=reward_cfg.get("use_similarity", True),
        use_props_delta=reward_cfg.get("use_props_delta", True),
        props_delta=reward_cfg.get("props_delta", {}),
        props_delta_penalty=reward_cfg.get("props_delta_penalty", {}),
        use_props_range=reward_cfg.get("use_props_range", True),
        props_range=reward_cfg.get("props_range", {}),
        props_range_penalty=reward_cfg.get("props_range_penalty", {}),
        scale=reward_cfg.get("reward_scale", 4.0),
        clip=reward_cfg.get("reward_clip", 1.0),
        admet_model=admet_model,
        task=reward_cfg.get("task", config.task or "plogp+qed"),
    )

    # -------------------------------------------------------------- Load Models #

    print("📦 [Model] Loading models and tokenizer...")
    actor_model, tokenizer = load_model_tokenizer(
        base_model=config.base_model,
        base_model_path=config.base_model_path,
        lora_adapter_path=config.lora_adapter_path,
        use_lora=config.use_lora,
        device=config.device,
    )
    ref_model = copy.deepcopy(actor_model).to('cpu')
    ref_model.eval()
    ref_model.requires_grad_(False)
    
    actor_model.config.use_cache = False
    ref_model.config.use_cache = False
    actor_model.enable_input_require_grads()

    actor = MoloActor(actor_model, tokenizer, device=config.device) 
    ref_actor = MoloActor(ref_model, tokenizer, device="cpu")
    optimizer_actor = torch.optim.AdamW(actor.parameters(), lr=config.lr_actor)
    adaptive_kl = AdaptiveKL(
        init_kl_coef=config.init_kl_coef, 
        target_kl=config.target_kl,
        alpha=config.kl_alpha,
        min_kl_coef=config.min_kl_coef, 
        max_kl_coef=config.max_kl_coef,
        )

    print("✅ [Model] Models and Tokenizer loaded.")

    # ----------------------------------------------------------- Build Envs & Datasets #

    print("📦 [Dataset] Building datasets...")
    train_ds, train_len, val_ds, val_len = build_datasets(config.dataset_cfg)
    dataset_cfg = config.dataset_cfg
    train_env_cfg = config.train_env_cfg
    eval_env_cfg = config.eval_env_cfg
    rl_task_cfg = config.rl_task_cfg
    print("📦 [Environment] Creating train/eval environments...")
    train_env = MoloEnv(
        dataset=train_ds,
        task=rl_task_cfg["task"],
        props_delta=rl_task_cfg["props_delta"],
        props_range=rl_task_cfg["props_range"],
        reward_weight=rl_task_cfg["reward_weight"],
        include_delta=train_env_cfg["include_delta_in_prompt"],
        include_weight=train_env_cfg["include_weight_in_prompt"],
        random_sample=train_env_cfg.get("random_sample_in_prompt", False),
        device=config.device,
    )
    eval_env = MoloEnv(
        dataset=val_ds,
        task=rl_task_cfg["task"],
        props_delta=rl_task_cfg["props_delta"],
        props_range=rl_task_cfg["props_range"],
        reward_weight=rl_task_cfg["reward_weight"],
        include_delta=eval_env_cfg["include_delta_in_prompt"],
        include_weight=eval_env_cfg["include_weight_in_prompt"],
        random_sample=eval_env_cfg.get("random_sample_in_prompt", False),
        device=config.device,
    )
    print(f"✅ [Environment]['train'] Ready with {train_len} samples.")
    print(f"✅ [Environment]['eval'] Ready with {val_len} samples.")
    buffer = GRPORolloutBuffer(train_env, tokenizer, device=config.device)
    

    trainer = GDPOTrainer(
        actor=actor,
        ref_actor=ref_actor,
        train_env=train_env,
        eval_env=eval_env,
        reward_fn=reward_fn,
        buffer=buffer,
        optimizer_actor=optimizer_actor,
        tokenizer=tokenizer,
        adaptive_kl=adaptive_kl,
        device=config.device,
        config=config,
    )
    trainer.train()
    if not args.skip_test_after_train:
        run_test_after_training(config=config, trainer=trainer, admet_model=admet_model)
    


if __name__ == "__main__":
    main()
