import torch
import torch.nn.functional as F


def compute_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits_shifted = logits[:, :-1, :]
    labels_shifted = labels[:, 1:]
    log_probs_all = F.log_softmax(logits_shifted, dim=-1)
    safe_labels = labels_shifted.clone()
    safe_labels[safe_labels == -100] = 0
    log_probs_selected = log_probs_all.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    log_probs_selected[labels_shifted == -100] = 0.0
    padded = torch.zeros_like(labels, dtype=log_probs_selected.dtype)
    padded[:, 1:] = log_probs_selected
    return padded
