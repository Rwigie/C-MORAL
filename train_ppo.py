import argparse
import torch
import copy
import os
import sys
from functools import partial

if __package__:
    from .rl import PPOTrainer, RolloutBuffer, AdaptiveKL
    from .rl.ppo.actor import MoloActor
    from .rl.ppo.critic import MoloCritic
    from .rl.reward import compute_reward
    from .utils import load_model_tokenizer, build_datasets, load_config, check_lora_weights
    from .rl.env import MoloEnv
else:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    from rl import PPOTrainer, RolloutBuffer, AdaptiveKL
    from rl.ppo.actor import MoloActor
    from rl.ppo.critic import MoloCritic
    from rl.reward import compute_reward
    from utils import load_model_tokenizer, build_datasets, load_config, check_lora_weights
    from rl.env import MoloEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to training config JSON/YAML.")
    return parser.parse_args()


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
        + f"Base Model      : {config.base_model}\n"
        + f"Model Path      : {config.base_model_path}\n"
        + f"LoRA Path       : {config.lora_adapter_path}\n"
        + f"Use LoRA        : {config.use_lora}\n"
        + f"Dataset Mode    : {dataset_cfg.get('mode', 'full')}\n"
        + f"Datasets        : {dataset_cfg.get('tasks', 'all')}\n"
        + f"RL Task         : {rl_task_cfg.get('task', config.task)}\n"
        + f"Actor lr        : {config.lr_actor}\n"
        + f"Critic lr       : {config.lr_critic}\n"
        + f"Init Kl coef    : {config.init_kl_coef}\n"
        + f"Reward Mode     : {rl_task_cfg.get('reward_mode', 'absolute')}\n"
        + f"Aggregation     : {rl_task_cfg.get('reward_aggregation', 'mean')}\n"
        + f"Use Similarity  : {rl_task_cfg.get('use_similarity', True)}\n"
        + f"Use Props Delta : {rl_task_cfg.get('use_props_delta', True)}\n"
        + f"Use Props Range : {rl_task_cfg.get('use_props_range', True)}\n"
        + f"num_iterations  : {config.num_iterations}\n"
        + f"ppo_epochs      : {config.ppo_epochs}\n"
        + f"batch_size      : {training_batch}\n"
        + f"mini_batch_size : {mini_batch}\n"
        + "=" * 80
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    configuration_plot(config)
    reward_cfg = dict(config.reward_cfg or config.rl_task_cfg or {})
    if config.task and "task" not in reward_cfg:
        reward_cfg["task"] = config.task

    reward_fn = partial(
        compute_reward,
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
        scale=reward_cfg.get("reward_scale", 1.0),
        clip=reward_cfg.get("reward_clip", 1.0),
        admet_model=reward_cfg.get("admet_model"),
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
    critic_model = actor_model if config.share_backbone else copy.deepcopy(actor_model)

    actor_model.config.use_cache = False
    critic_model.config.use_cache = False
    ref_model.config.use_cache = False
    actor_model.enable_input_require_grads()
    critic_model.enable_input_require_grads()

    actor = MoloActor(actor_model, tokenizer, device=config.device)
    
    ref_actor = MoloActor(ref_model, tokenizer, device="cpu")
    critic = MoloCritic(critic_model, device=config.device)
    adaptive_kl = AdaptiveKL(
        target_kl=config.target_kl,
        init_kl_coef=config.init_kl_coef,
        alpha=config.kl_alpha,
        min_kl_coef=config.min_kl_coef,
        max_kl_coef=config.max_kl_coef,
    )
    optimizer_actor = torch.optim.AdamW(actor.parameters(), lr=config.lr_actor)
    optimizer_critic = torch.optim.AdamW(critic.parameters(), lr=config.lr_critic)

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
        device=config.device,
        random_sample=train_env_cfg.get("random_sample_in_prompt", False),
    )
    eval_env = MoloEnv(
        dataset=val_ds,
        task=rl_task_cfg["task"],
        props_delta=rl_task_cfg["props_delta"],
        props_range=rl_task_cfg["props_range"],
        reward_weight=rl_task_cfg["reward_weight"],
        include_delta=eval_env_cfg["include_delta_in_prompt"],
        include_weight=eval_env_cfg["include_weight_in_prompt"],
        device=config.device,
        random_sample=eval_env_cfg.get("random_sample_in_prompt", False),
    )
    print(f"✅ [Environment]['train'] Ready with {train_len} samples.")
    print(f"✅ [Environment]['eval'] Ready with {val_len} samples.")
    buffer = RolloutBuffer(train_env, tokenizer, device=config.device)
    

    trainer = PPOTrainer(
        actor=actor,
        ref_actor=ref_actor,
        critic=critic,
        train_env=train_env,
        eval_env=eval_env,
        reward_fn=reward_fn,
        buffer=buffer,
        optimizer_actor=optimizer_actor,
        optimizer_critic=optimizer_critic,
        tokenizer=tokenizer,
        adaptive_kl=adaptive_kl,
        device=config.device,
        config=config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
