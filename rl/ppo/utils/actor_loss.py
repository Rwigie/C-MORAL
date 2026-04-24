import torch

from ...utils import compute_log_probs

from .approx_kl import compute_approx_kl


def compute_actor_loss(actor, old_log_probs, ref_log_probs, input_ids, attention_mask, advantages, labels, clip_eps, adv_norm=False):
    mask = (labels != -100).float()
    valid_advs = torch.masked_select(advantages, labels != -100)

    if adv_norm:
        adv_mean = valid_advs.mean()
        adv_std = valid_advs.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        advantages = advantages * mask

    _, logits = actor(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    new_log_probs = compute_log_probs(logits, labels)

    log_ratio = (new_log_probs - old_log_probs) * mask
    ratio = torch.exp(log_ratio)
    # print(f"[DEBUG] ratio stats: mean={ratio.mean().item():.6f}, std={ratio.std().item():.6f}")
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    token_loss = -torch.min(surrogate1, surrogate2)
    policy_loss = token_loss.sum() / mask.sum().clamp(min=1.0)

    with torch.no_grad():
        kl_div = compute_approx_kl(new_log_probs, ref_log_probs, mask)
        mean_kl = kl_div.sum() / mask.sum().clamp(min=1.0)
    return policy_loss, mean_kl
