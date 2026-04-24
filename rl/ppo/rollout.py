from dataclasses import dataclass
from typing import Dict, List
import inspect

import torch
from torch.nn.utils.rnn import pad_sequence

from ..utils import compute_log_probs


def _call_reward_fn_with_targets(reward_fn, source_smiles, target_smiles, target_thresholds_batch):
    try:
        sig = inspect.signature(reward_fn)
        if "target_thresholds_batch" in sig.parameters:
            return reward_fn(
                source_smiles,
                target_smiles,
                target_thresholds_batch=target_thresholds_batch,
            )
    except (TypeError, ValueError):
        pass
    return reward_fn(source_smiles, target_smiles)


def _is_llama_actor(actor) -> bool:
    """
    Best-effort model family detection for runtime memory policy.
    """
    model = getattr(actor, "model", None)
    if model is None:
        return False

    cfg = getattr(model, "config", None)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    if "llama" in model_type:
        return True

    base_model = getattr(model, "base_model", None)
    base_cfg = getattr(base_model, "config", None)
    base_model_type = str(getattr(base_cfg, "model_type", "")).lower()
    return "llama" in base_model_type


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    scores: List[List[Dict]]
    masking_mask: torch.Tensor
    source_smiles: List[str]
    prompt: List[str]
    prompt_instr: Dict[str, torch.Tensor]



class RolloutBuffer:
    """
    Collects rollouts from the actor and caches all tensors required by PPO updates.
    Rewritten as a reusable component instead of ad-hoc logic inside `train.py`.
    """
    def __init__(self, env, tokenizer, device="cpu"):
        self.env = env
        self.tokenizer = tokenizer
        self.device = device
        self.clear()
        self.clear_prompt()

    # -------------------------- collection helpers -------------------------- #

    def clear(self):
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        self.log_probs = []
        self.ref_log_probs = []
        self.values = []
        self.rewards = []
        self.scores = []

    def clear_prompt(self):
        self.source_smiles = []
        self.prompt = []
        self.prompt_instr = []
        self.target_thresholds_batch = []

    def batch_padding(self, sequences):
        """
        Pad list of tensors to a dense batch according to tokenizer settings.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        return pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

    def _calculate_log_probs(self, actor, input_ids, labels, attention_mask):
        with torch.no_grad():
            if not _is_llama_actor(actor):
                _, logits = actor(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                return compute_log_probs(logits, labels)

            chunk_size = 128
            total = input_ids.size(0)
            chunks = []
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                _, logits = actor(
                    input_ids=input_ids[start:end],
                    attention_mask=attention_mask[start:end],
                    labels=labels[start:end],
                )
                chunks.append(compute_log_probs(logits, labels[start:end]))
            return torch.cat(chunks, dim=0)

    def _calculate_values(self, critic, input_ids, attention_mask):
        with torch.no_grad():
            return critic(input_ids=input_ids, attention_mask=attention_mask)

    # ------------------------------ public API ------------------------------ #

    def collect(
        self,
        actor,
        ref_actor,
        critic,
        reward_fn,
        num_return_sequences: int = 2,
        num_envs: int = 2,
        do_sample: bool = True,
        temperature: float = 0.8,
        max_new_tokens: int = 100,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        """
        Collect a full PPO batch by sampling the actor and querying the critic.
        """
        self.clear()
        self.clear_prompt()

        prompt, source_smiles, task, done = self.env.reset()

        for i in range(num_envs):
            self.prompt.append(prompt)
            self.source_smiles.append(source_smiles)
            self.target_thresholds_batch.append(dict(getattr(self.env, "current_target_thresholds", {}) or {}))
            if done:
                prompt, source_smiles, task, done = self.env.reset()
            else:
                prompt, source_smiles, task, done = self.env.step()

        with torch.no_grad():
            input_ids, attention_mask, labels, target_smiles, prompt_instr = actor.generate(
                prompt=self.prompt,
                num_envs=num_envs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
            )

        rewards_dict = _call_reward_fn_with_targets(
            reward_fn,
            self.source_smiles,
            target_smiles,
            self.target_thresholds_batch,
        )
        rewards = rewards_dict["reward"].to(self.device)
        scores = rewards_dict["scores"].to(self.device)
        ref_actor = ref_actor.to(self.device)
        ref_log_probs = self._calculate_log_probs(actor=ref_actor, input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        log_probs = self._calculate_log_probs(actor=actor, input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        values = self._calculate_values(critic=critic, input_ids=input_ids, attention_mask= attention_mask)
        ref_actor = ref_actor.to("cpu")

        self.input_ids = input_ids.cpu()
        self.attention_mask = attention_mask.cpu()
        self.labels = labels.cpu()
        self.rewards = rewards.cpu()
        self.prompt_instr = prompt_instr
        self.log_probs = log_probs.cpu()
        self.ref_log_probs = ref_log_probs.cpu()
        self.values = values.cpu()
        self.scores = scores.cpu()

    def get(self) -> RolloutBatch:
        """
        Return a snapshot of the most recent rollout batch.
        """
        return RolloutBatch(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            log_probs=self.log_probs,
            ref_log_probs=self.ref_log_probs,
            values=self.values,
            rewards=self.rewards,
            source_smiles=list(self.source_smiles),
            prompt=list(self.prompt),
            prompt_instr=self.prompt_instr,
        )
    
class GRPORolloutBuffer:
    """
    Collects rollouts from the actor and caches all tensors required by PPO updates.
    Rewritten as a reusable component instead of ad-hoc logic inside `train.py`.
    """

    def __init__(self, env, tokenizer, device="cpu"):
        self.env = env
        self.tokenizer = tokenizer
        self.device = device
        self.clear()
        self.clear_prompt()

    # -------------------------- collection helpers -------------------------- #

    def clear(self):
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        self.log_probs = []
        self.ref_log_probs = []
        self.rewards = []
        self.scores = []
        self.masking_mask = []
        self.target_properties = []

    def clear_prompt(self):
        self.source_smiles = []
        self.prompt = []
        self.prompt_instr = []
        self.target_thresholds_batch = []

    def batch_padding(self, sequences):
        """
        Pad list of tensors to a dense batch according to tokenizer settings.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        return pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

    def _calculate_log_probs(self, actor, input_ids, labels, attention_mask):
        with torch.no_grad():
            if not _is_llama_actor(actor):
                _, logits = actor(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                return compute_log_probs(logits, labels)

            chunk_size = 128
            total = input_ids.size(0)
            chunks = []
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                _, logits = actor(
                    input_ids=input_ids[start:end],
                    attention_mask=attention_mask[start:end],
                    labels=labels[start:end],
                )
                chunks.append(compute_log_probs(logits, labels[start:end]))
            return torch.cat(chunks, dim=0)


    # ------------------------------ public API ------------------------------ #

    def collect(
        self,
        actor,
        ref_actor,
        reward_fn,
        num_return_sequences: int = 2,
        num_envs: int = 2,
        do_sample: bool = True,
        temperature: float = 0.8,
        max_new_tokens: int = 100,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        """
        Collect a full PPO batch by sampling the actor and querying the critic.
        """
        self.clear()
        self.clear_prompt()

        prompt, source_smiles, task, done = self.env.reset()

        for i in range(num_envs):
            self.prompt.append(prompt)
            self.source_smiles.append(source_smiles)
            self.target_thresholds_batch.append(dict(getattr(self.env, "current_target_thresholds", {}) or {}))
            if done:
                prompt, source_smiles, task, done = self.env.reset()
            else:
                prompt, source_smiles, task, done = self.env.step()

        with torch.no_grad():
            input_ids, attention_mask, labels, target_smiles, prompt_instr = actor.generate(
                prompt=self.prompt,
                num_envs=num_envs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
            )

        reward_dict = _call_reward_fn_with_targets(
            reward_fn,
            self.source_smiles,
            target_smiles,
            self.target_thresholds_batch,
        )
        rewards = reward_dict["reward"]
        scores = reward_dict["scores"]
        masking_mask = reward_dict["valid_mask"]
        target_properties = reward_dict.get("target_properties", [])
        ref_actor = ref_actor.to(self.device)
        ref_log_probs = self._calculate_log_probs(actor=ref_actor, input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        log_probs = self._calculate_log_probs(actor=actor, input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        ref_actor = ref_actor.to("cpu")

        self.input_ids = input_ids.cpu()
        self.attention_mask = attention_mask.cpu()
        self.labels = labels.cpu()
        self.rewards = rewards.cpu()
        self.scores = {k: v.to('cpu') for k, v in scores.items()}
        self.masking_mask = masking_mask.cpu()
        self.target_properties = target_properties
        self.prompt_instr = prompt_instr
        self.log_probs = log_probs.cpu()
        self.ref_log_probs = ref_log_probs.cpu()

    def get(self) -> RolloutBatch:
        """
        Return a snapshot of the most recent rollout batch.
        """
        return RolloutBatch(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            log_probs=self.log_probs,
            ref_log_probs=self.ref_log_probs,
            values=torch.empty(0),
            rewards=self.rewards,
            source_smiles=list(self.source_smiles),
            prompt=list(self.prompt),
            prompt_instr=self.prompt_instr,
            scores = self.scores,
            masking_mask = self.masking_mask,
        )
