import os
from datetime import datetime
import math

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import wandb

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore

from ...utils import (
    log_metrics,
    save_logs,
    save_checkpoint,
    save_best_adapter,
    check_lora_weights,
    compute_property_mean,
    compute_score_mean
)
from .utils import (
    compute_actor_loss,
    compute_critic_loss,
    compute_gae_advantages_returns,
)
from .rollout import RolloutBuffer
from ...eval import eval_train_env


class PPOTrainer:
    """
    High-level orchestrator that glues together rollout collection,
    PPO updates, evaluation, and logging.
    """

    def __init__(
        self,
        actor,
        ref_actor,
        critic,
        train_env,
        eval_env,
        reward_fn,
        buffer: RolloutBuffer,
        optimizer_actor,
        optimizer_critic,
        tokenizer,
        adaptive_kl,
        joint_optimizer=None,
        device: str = "cpu",
        config=None,
    ):
        self.actor = actor
        self.ref_actor = ref_actor
        self.critic = critic
        self.train_env = train_env
        self.eval_env = eval_env
        self.reward_fn = reward_fn
        self.buffer = buffer
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.joint_optimizer = joint_optimizer
        self.tokenizer = tokenizer
        self.adaptive_kl = adaptive_kl
        self.device = device
        if config is None:
            raise ValueError("config must be provided to PPOTrainer.")
        self.config = config

        self.actor_params = [p for p in self.actor.model.parameters() if p.requires_grad]
        self.critic_params = list(self.critic.value_head.parameters())

        self.best_reward = float("-inf")
        self.run_dirs = {}
        self.tb_writer = None
        self.logging_backends = []
        self.actor_scheduler = None
        self.critic_scheduler = None

    # ------------------------------------------------------------------ setup #

    def _prepare_run_dirs(self):
        cfg = self.config
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_dir = os.path.join(cfg.runs_dir, run_id)
        checkpoints_dir = os.path.join(runs_dir, cfg.checkpoints_dir)
        logs_dir = os.path.join(runs_dir, "logs")
        best_adapter_dir = os.path.join(runs_dir, cfg.best_adapter_dir)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(best_adapter_dir, exist_ok=True)
        self.run_dirs = {
            "runs_dir": runs_dir,
            "checkpoints_dir": checkpoints_dir,
            "logs_dir": logs_dir,
            "best_adapter_dir": best_adapter_dir,
        }

    def _init_wandb(self):
        cfg = self.config
        if not cfg.use_wandb:
            return
        wandb_run_name = cfg.wandb_run_name or f"run_{self.run_dirs['runs_dir'].split('/')[-1]}"
        config = {
            "num_iterations": cfg.num_iterations,
            "ppo_epochs": cfg.ppo_epochs,
            "batch_size": cfg.num_return_sequences * cfg.num_envs,
            "num_mini_batch": cfg.num_mini_batch,
            "clip_eps": cfg.clip_eps,
            "gamma": cfg.gamma,
            "lr_actor": cfg.lr_actor,
            "lr_critic": cfg.lr_critic,
            "value_clip": cfg.value_clip,
            "temperature": cfg.temperature,
            "max_new_tokens": cfg.max_new_tokens,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "init_kl_coef": cfg.init_kl_coef,
            "target_kl": cfg.target_kl,
            "min_kl_coef": cfg.min_kl_coef,
            "max_kl_coef": cfg.max_kl_coef,
            "train_do_sample": cfg.train_do_sample,
            "eval_do_sample": cfg.eval_do_sample,
            "use_sequence_value": cfg.use_sequence_value,

        }
        if cfg.wandb_config:
            config.update(cfg.wandb_config)

        wandb.init(
            project=cfg.wandb_project,
            name=wandb_run_name,
            config=config,
            dir=self.run_dirs["runs_dir"],
        )

    def _init_tensorboard(self):
        backend = (self.config.logging_backend or "").lower()
        if backend not in ("tensorboard", "both"):
            return
        if SummaryWriter is None:
            print("[Warn] TensorBoard not available; skipping tensorboard logging.")
            return
        tb_dir = os.path.join(self.run_dirs["runs_dir"], "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tb_dir)

    def _resolve_logging_backends(self):
        backend = (self.config.logging_backend or "none").lower()
        mapping = {
            "wandb": ["wandb"],
            "tensorboard": ["tensorboard"],
            "both": ["wandb", "tensorboard"],
            "none": [],
        }
        backends = mapping.get(backend, [])
        resolved = []
        for name in backends:
            if name == "wandb" and not self.config.use_wandb:
                continue
            if name == "tensorboard" and self.tb_writer is None:
                continue
            resolved.append(name)
        self.logging_backends = resolved

    def _init_schedulers(self):
        cfg = self.config
        use_sched = bool(getattr(cfg, "use_lr_scheduler", False))
        sched_type = str(getattr(cfg, "lr_scheduler_type", "cosine")).lower()
        if not use_sched or sched_type in {"none", "off", "false"}:
            self.actor_scheduler = None
            self.critic_scheduler = None
            return

        total_steps = max(1, int(getattr(cfg, "num_iterations", 1)))
        warmup_ratio = float(getattr(cfg, "warmup_ratio", 0.0))
        warmup_steps = int(total_steps * max(0.0, min(1.0, warmup_ratio)))

        def build_lambda(init_lr: float, min_lr: float):
            min_factor = (min_lr / init_lr) if init_lr > 0 else 0.0
            min_factor = max(0.0, min(1.0, min_factor))

            def lr_lambda(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)

                if total_steps <= warmup_steps:
                    base = 1.0
                else:
                    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    progress = max(0.0, min(1.0, progress))
                    if sched_type == "linear":
                        base = 1.0 - progress
                    else:
                        base = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * max(0.0, base)

            return lr_lambda

        actor_init_lr = float(getattr(cfg, "lr_actor", 0.0) or 0.0)
        actor_min_lr = float(getattr(cfg, "min_lr_actor", 0.0) or 0.0)
        critic_init_lr = float(getattr(cfg, "lr_critic", 0.0) or 0.0)
        critic_min_lr = float(getattr(cfg, "min_lr_critic", 0.0) or 0.0)

        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_actor, lr_lambda=build_lambda(actor_init_lr, actor_min_lr)
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_critic, lr_lambda=build_lambda(critic_init_lr, critic_min_lr)
        )

    # -------------------------------------------------------------- utilities #

    def _toggle_gradient_checkpointing(self, enable: bool):
        """Mirror the original logic for enabling/disabling checkpointing."""
        fn_name = "gradient_checkpointing_enable" if enable else "gradient_checkpointing_disable"
        model = self.actor.model
        if hasattr(model, fn_name):
            getattr(model, fn_name)()
        elif hasattr(model, "base_model") and hasattr(model.base_model, fn_name):
            getattr(model.base_model, fn_name)()
        if hasattr(self.critic, "set_gradient_checkpointing"):
            self.critic.set_gradient_checkpointing(enable)

    # ----------------------------------------------------------- main runner #

    def train(self):
        self._prepare_run_dirs()
        self._init_wandb()
        self._init_tensorboard()
        self._resolve_logging_backends()
        self._init_schedulers()

        print("[Eval] Evaluating base model before RL training...")
        self.eval_env.reset()
        self.evaluate(iteration=0, verbose=True)

        try:
            for iteration in range(self.config.num_iterations):
                print(f"\n{'-' * 50}\n>> Iteration {iteration} / {self.config.num_iterations}\n{'-' * 50}")
                rollout_stats = self.collect_rollouts()

                metrics = self._run_ppo_epochs()
                self.buffer.clear()

                train_metrics = {
                    "iteration": iteration,
                    "mean": rollout_stats["mean"],
                    "std": rollout_stats["std"],
                    "min": rollout_stats["min"],
                    "max": rollout_stats["max"],
                    "actor_loss": metrics["actor_loss_mean"],
                    "critic_loss": metrics["critic_loss_mean"],
                    "kl_loss": metrics["kl_loss_mean"],
                    "kl_coef": self.adaptive_kl.kl_coef,
                    "lr_actor": float(self.optimizer_actor.param_groups[0]["lr"]),
                    "lr_critic": float(self.optimizer_critic.param_groups[0]["lr"]),
                }
                log_metrics(train_metrics, step=iteration, mode="train", backends=self.logging_backends, writer=self.tb_writer)
                save_logs(train_metrics, log_dir=self.run_dirs["logs_dir"])

                print(
                    "[Train]\n"
                    f"  Actor Loss   : {metrics['actor_loss_mean']:.4f}\n"
                    f"  Critic Loss  : {metrics['critic_loss_mean']:.4f}\n"
                    f"  Mean Reward  : {rollout_stats['mean']:.4f}\n"
                    f"  KL Loss      : {metrics['kl_loss_mean']:.4f}"
                )

                if iteration % self.config.eval_interval == 0 and iteration != 0:
                    self.evaluate(iteration, verbose=True)
                if iteration % self.config.save_interval == 0:
                    self._save_checkpoint(iteration)
                if self.actor_scheduler is not None:
                    self.actor_scheduler.step()
                if self.critic_scheduler is not None:
                    self.critic_scheduler.step()

        finally:
            if self.tb_writer is not None:
                self.tb_writer.close()

    # --------------------------------------------------------- core routines #

    def collect_rollouts(self):
        print("[Rollout] Collecting samples... ")
        self.actor.model.train()
        self.critic.value_head.train()
        self._toggle_gradient_checkpointing(enable=False)
        self.buffer.collect(
            actor=self.actor,
            ref_actor=self.ref_actor,
            critic=self.critic,
            reward_fn=self.reward_fn,
            num_return_sequences=self.config.num_return_sequences,
            num_envs=self.config.num_envs,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            do_sample=self.config.train_do_sample,
        )
        stats = {
            "mean": self.buffer.rewards.mean().item(),
            "std": self.buffer.rewards.std().item(),
            "min": self.buffer.rewards.min().item(),
            "max": self.buffer.rewards.max().item(),
        }
        return stats

    # -------------------------------------------------------- ppo training #

    def _run_ppo_epochs(self):
        print("[Train] Enabling gradient checkpointing for training...")
        self._toggle_gradient_checkpointing(enable=True)
        print("[Train] Start PPO training...")
        cfg = self.config

        input_ids = self.buffer.input_ids
        attention_mask = self.buffer.attention_mask
        labels = self.buffer.labels
        old_log_probs = self.buffer.log_probs
        ref_log_probs = self.buffer.ref_log_probs
        values = self.buffer.values

        batch_size = input_ids.size(0)
        mini_batch_size = batch_size // cfg.num_mini_batch if cfg.num_mini_batch != 0 else batch_size
        num_mini_batch = cfg.num_mini_batch if cfg.num_mini_batch <= batch_size else 1
        if batch_size % num_mini_batch != 0:
            raise ValueError(
                f"[Error] batch_size ({batch_size}) must be divisible by num_mini_batch ({num_mini_batch})."
            )

        actor_losses = []
        critic_losses = []
        kl_losses = []

        advantages, returns = compute_gae_advantages_returns(
            seq_rewards=self.buffer.rewards,
            values=self.buffer.values,
            labels=self.buffer.labels,
            old_log_probs=self.buffer.log_probs,
            ref_log_probs=self.buffer.ref_log_probs,
            kl_coef=self.adaptive_kl.kl_coef,
            gamma=self.config.gamma,
            lam=self.config.lam,
            use_sequence_value=self.config.use_sequence_value,
        )

        for epoch in tqdm(range(cfg.ppo_epochs), desc="PPO Epochs"):
            indices = torch.randperm(batch_size)
            epoch_kl = 0.0

            for i in range(num_mini_batch):
                start = i * mini_batch_size
                end = start + mini_batch_size
                mb_idx = indices[start:end]

                mb_input_ids = input_ids[mb_idx].to(self.device)
                mb_attention_mask = attention_mask[mb_idx].to(self.device)
                mb_labels = labels[mb_idx].to(self.device)
                mb_old_log_probs = old_log_probs[mb_idx].to(self.device)
                mb_ref_log_probs = ref_log_probs[mb_idx].to(self.device)
                mb_advantages = advantages[mb_idx].to(self.device)
                mb_returns = returns[mb_idx].to(self.device)
                mb_old_values = values[mb_idx].to(self.device)

                policy_loss, critic_loss, mb_kl = self.update(
                    mb_input_ids=mb_input_ids,
                    mb_attention_mask=mb_attention_mask,
                    mb_labels=mb_labels,
                    mb_old_log_probs=mb_old_log_probs,
                    mb_ref_log_probs=mb_ref_log_probs,
                    mb_advantages=mb_advantages,
                    mb_returns=mb_returns,
                    mb_old_values=mb_old_values,
                )
                if policy_loss is None:
                    continue

                actor_losses.append(policy_loss)
                critic_losses.append(critic_loss)
                kl_losses.append(mb_kl)
                epoch_kl += mb_kl
                self.adaptive_kl.update(mb_kl)
                
            avg_kl = epoch_kl / max(1, num_mini_batch)
            if avg_kl > self.config.early_stop_kl:
                print(f"Early stop at epoch {epoch}, KL={avg_kl:.4f}")
                break
            

        return {
            "actor_loss_mean": sum(actor_losses) / max(1, len(actor_losses)),
            "critic_loss_mean": sum(critic_losses) / max(1, len(critic_losses)),
            "kl_loss_mean": sum(kl_losses) / max(1, len(kl_losses)),
        }

    def _save_checkpoint(self, iteration: int):
        save_checkpoint(
            actor=self.actor,
            optimizer_actor=self.optimizer_actor,
            critic=self.critic,
            optimizer_critic=self.optimizer_critic,
            save_dir=self.run_dirs["checkpoints_dir"],
            iteration=iteration,
            save_full_critic=self.config.save_full_critic,
        )

    # ------------------------------------------------------ single update step #
    
    def update(
        self,
        mb_input_ids,
        mb_attention_mask,
        mb_labels,
        mb_old_log_probs,
        mb_ref_log_probs,
        mb_advantages,
        mb_returns,
        mb_old_values,
    ):
        cfg = self.config
        if self.config.joint_update and self.joint_optimizer is None:
            raise ValueError("joint_update=True but joint_optimizer is not provided.")

        if cfg.joint_update is False:
            self.actor.model.train()
            policy_loss, mb_kl = compute_actor_loss(
                actor=self.actor,
                old_log_probs=mb_old_log_probs,
                ref_log_probs=mb_ref_log_probs,
                input_ids=mb_input_ids,
                attention_mask=mb_attention_mask,
                advantages=mb_advantages.detach(),
                labels=mb_labels,
                clip_eps=cfg.clip_eps,
            )
            if not torch.isfinite(policy_loss):
                print("[WARN] policy_loss is NaN/Inf, skip this batch.")
                return None, None, None
            self.optimizer_actor.zero_grad()
            policy_loss.backward()
            clip_grad_norm_(self.actor_params, max_norm=0.5)
            self.optimizer_actor.step()
            policy_loss_value = policy_loss.item()
            del policy_loss

            self.critic.backbone.train()
            self.critic.value_head.train()
            critic_loss = compute_critic_loss(
                critic=self.critic,
                input_ids=mb_input_ids,
                attention_mask=mb_attention_mask,
                labels=mb_labels,
                returns=mb_returns,
                old_values=mb_old_values,
                value_clip=cfg.value_clip,
                use_sequence_value=cfg.use_sequence_value,
            )
            if not torch.isfinite(critic_loss):
                print("[WARN] critic_loss is NaN/Inf, skip this batch.")
                return None, None, None
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic_params, max_norm=1.0)
            self.optimizer_critic.step()
            critic_loss_value = critic_loss.item()
            del critic_loss        
        else:
            total_loss = policy_loss + cfg.critic_loss_coef * critic_loss
            self.joint_optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(self.actor_params, max_norm=0.5)
            clip_grad_norm_(self.critic_params, max_norm=1.0)
            self.joint_optimizer.step()
            

            

        return policy_loss_value, critic_loss_value, mb_kl.item()

    # ------------------------------------------------------------- evaluation #
    @torch.no_grad()
    def evaluate(self, iteration: int, verbose: bool = False):
        chunk_size = 128
        eval_num_envs = int(self.config.eval_num_envs)
        if eval_num_envs <= chunk_size:
            result = eval_train_env(
                actor=self.actor,
                env=self.eval_env,
                tokenizer=self.tokenizer,
                reward_fn=self.reward_fn,
                num_return_sequences=self.config.eval_num_return_sequences,
                num_envs=eval_num_envs,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=self.config.eval_do_sample,
            )
        else:
            reward_chunks = []
            target_properties = []
            for start in range(0, eval_num_envs, chunk_size):
                end = min(start + chunk_size, eval_num_envs)
                chunk_indices = list(range(start, end))
                chunk_result = eval_train_env(
                    actor=self.actor,
                    env=self.eval_env,
                    tokenizer=self.tokenizer,
                    reward_fn=self.reward_fn,
                    num_return_sequences=self.config.eval_num_return_sequences,
                    num_envs=len(chunk_indices),
                    temperature=self.config.temperature,
                    max_new_tokens=self.config.max_new_tokens,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    do_sample=self.config.eval_do_sample,
                    eval_indices=chunk_indices,
                )
                reward_chunks.append(chunk_result["rewards"].detach().cpu())
                target_properties.extend(chunk_result.get("target_properties", []))
            result = {
                "rewards": torch.cat(reward_chunks, dim=0) if reward_chunks else torch.tensor([]),
                "target_properties": target_properties,
            }

        rewards = result["rewards"]
        eval_reward_mean = rewards.mean().item()
        eval_reward_std = rewards.std().item()
        eval_reward_min = rewards.min().item()
        eval_reward_max = rewards.max().item()
        task_keys = [k.strip().lower() for k in (self.config.task or "").split("+") if k.strip()]
        keys = list(dict.fromkeys([*task_keys, "sim"]))
        metrics = {"reward_mean": eval_reward_mean, **compute_target_property_means(result, self.config.task)}
        if verbose:
            print(
                "[Eval]\n"
                f"  reward_mean: {eval_reward_mean:.4f}\n"
                f"  reward_std : {eval_reward_std:.4f}\n"
                f"  reward_min : {eval_reward_min:.4f}\n"
                f"  reward_max : {eval_reward_max:.4f}"
            )
            for k in keys:
                v = metrics.get(k)
                if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                    print(f"  {k}: nan")
                else:
                    print(f"  {k}: {v:.4f}")
            metrics.update(
                {
                    "reward_std": eval_reward_std,
                    "reward_min": eval_reward_min,
                    "reward_max": eval_reward_max,
                }
            )
        log_metrics(metrics, step=iteration, mode="eval", backends=self.logging_backends, writer=self.tb_writer)
        save_logs({"iteration": iteration, **metrics}, log_dir=self.run_dirs["logs_dir"], filename="eval_logs.jsonl")
        if eval_reward_mean > self.best_reward:
            self.best_reward = eval_reward_mean
            if hasattr(self.actor, "save_pretrained"):
                save_best_adapter(self.actor, self.run_dirs["best_adapter_dir"])
        return eval_reward_mean
