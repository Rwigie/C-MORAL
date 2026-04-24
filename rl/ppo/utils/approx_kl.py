import torch


def compute_approx_kl_k3(log_probs, ref_log_probs, mask=None):
    """
    Estimate the KL divergence between the current policy and reference policy.
    Supports optional masking to ignore padding tokens.
    """
    with torch.no_grad():
        log_ratio = log_probs - ref_log_probs
        ratio = torch.exp(log_ratio)
        approx_kl = ratio - 1.0 - log_ratio
        if mask is not None:
            approx_kl = approx_kl * mask
    return approx_kl

def compute_approx_kl(log_probs, ref_log_probs, mask=None):
    with torch.no_grad():
        approx_kl = log_probs - ref_log_probs
        if mask is not None:
            approx_kl = approx_kl * mask
    return approx_kl
