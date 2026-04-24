from typing import Tuple

import torch

from .approx_kl import compute_approx_kl


@torch.no_grad()
def compute_gae_advantages_returns(
    seq_rewards,
    values,
    labels,
    old_log_probs,
    ref_log_probs,
    kl_coef,
    gamma: float = 0.99,
    lam: float = 0.95,
    use_sequence_value: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute token-level advantages and returns.
    Falls back to a sequence-value formulation when `use_sequence_value=True`.
    """
    device = values.device
    values = values.detach()
    seq_rewards = seq_rewards.to(device)
    B, T = labels.shape
    if values.dim() == 2 and values.size(1) == 1 and T > 1:
        values = values.expand(-1, T)
    elif values.dim() != 2 or values.size(1) != T:
        raise ValueError(f"Shape mismatch: values {values.shape} vs labels {labels.shape}")

    mask = (labels != -100).float()
    kl = compute_approx_kl(old_log_probs, ref_log_probs, mask)
    rewards = -kl_coef * kl

    if use_sequence_value:
        denom = mask.sum(dim=1).clamp(min=1.0)
        kl_mean = (kl * mask).sum(dim=1) / denom
        seq_reward_adv = seq_rewards - kl_coef * kl_mean
        seq_reward_ret = seq_rewards

        lengths = mask.sum(dim=1).long() - 1
        lengths = torch.clamp(lengths, min=0)

        if values.size(1) == 1:
            seq_values = values.squeeze(1)
        else:
            seq_values = torch.zeros(B, device=device, dtype=values.dtype)
            batch_idx = torch.arange(B, device=device)
            valid = mask.sum(dim=1) > 0
            # For using the value at the last valid token of the prompt+response Q(s,a)
            # seq_values[valid] = values[batch_idx[valid], lengths[valid]]

            # For using the value at the last token of the prompt V(s)
            response_start_indices = mask.argmax(dim=1)
            prompt_end_indices = response_start_indices - 1
            prompt_end_indices = prompt_end_indices.clamp(min=0)
            seq_values[valid] = values[batch_idx[valid], prompt_end_indices[valid]]

        delta = seq_reward_adv - seq_values
        adv = delta.unsqueeze(1) * mask
        ret = seq_reward_ret
        return adv.detach(), ret.detach()

    # distribute sequence reward to last valid token
    for b in range(B):
        idxs = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            continue
        last_t = idxs[-1].item()
        rewards[b, last_t] += seq_rewards[b]

    lastgaelam = 0
    adv_rev = []
    for t in reversed(range(T)):
        nextvalues = values[:, t + 1] if t < T - 1 else 0.0
        next_mask = mask[:, t + 1] if t < T - 1 else 0.0
        nextvalues = nextvalues * next_mask
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = (delta + gamma * lam * lastgaelam) * mask[:, t]
        adv_rev.append(lastgaelam)

    advantages = torch.stack(adv_rev[::-1], dim=1)
    returns = advantages + values
    return advantages, returns

@torch.no_grad()
def compute_grpo_advantages(
    num_envs: int,
    num_return_sequences: int,
    rewards: torch.Tensor = None,
    labels: torch.Tensor = None,
    old_log_probs: torch.Tensor = None,
    ref_log_probs: torch.Tensor = None,
    kl_coef: float = 0.0,
    epsilon: float = 1e-8,
    episilon: float = None,
    use_sequence_value: bool = True,
    adv_norm: bool = True,
):
    """
    Compute GRPO (group-relative) advantages.

    This is typically used with a sequence-level reward `R` and then broadcast to all response tokens.
    Optionally includes a KL penalty via `old_log_probs`/`ref_log_probs` and `kl_coef`.
    Args:
        rewards/seq_rewards: Tensor of shape (B,) where B = num_envs * num_return_sequences
        num_envs: Number of parallel environments
        num_return_sequences: Number of return sequences per environment
        epsilon: Small value to avoid division by zero
    Returns:
        advantages: Tensor of shape (B, T) when `labels` is provided, else (B,)
     """

    if episilon is not None:
        epsilon = episilon

    device = rewards.device
    rewards = rewards.to(device)

    if use_sequence_value:
        seq_rewards = rewards

        mask = (labels != -100).float().to(device)
        kl = compute_approx_kl(old_log_probs.to(device), ref_log_probs.to(device), mask)
        denom = mask.sum(dim=1).clamp(min=1.0)
        kl_mean = (kl * mask).sum(dim=1) / denom
        #adjusted_rewards = seq_rewards - kl_coef * kl_mean
        adjusted_rewards = seq_rewards


        group_rewards = adjusted_rewards.view(num_envs, num_return_sequences)
        group_mean = group_rewards.mean(dim=1, keepdim=True)
        group_std = group_rewards.std(dim=1, keepdim=True, unbiased=False)
        seq_adv = (group_rewards - group_mean) / (group_std + epsilon)
        seq_adv = seq_adv.view(-1)

        if labels is None:
            return seq_adv

        mask = (labels != -100).float().to(device)
        adv = seq_adv.unsqueeze(1) * mask
        return adv.detach()


@torch.no_grad()
def compute_gdpo_advantages(
    rewards: torch.Tensor,  # shape (B,)
    labels: torch.Tensor,      # shape (B, T)
):
    if len(rewards) > 1:
        rewards_mean = rewards.mean()
        rewards_std = rewards.std(unbiased=False)
        rewards = (rewards - rewards_mean) / (rewards_std + 1e-8)
        
    mask = (labels != -100).float()
    adv = rewards.unsqueeze(1) * mask  # (B,1) -> (B,T)
    return adv.detach()


