import json
import os
from typing import Iterable, Optional, Union

import wandb

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore


def _format_metrics(metrics: dict, mode: Optional[str]) -> dict:
    if not mode:
        return metrics
    return {f"{mode}/{k}": v for k, v in metrics.items()}


def log_metrics(
    metrics: dict,
    step: int = None,
    mode: str = "train",
    backends: Union[str, Iterable[str]] = "wandb",
    writer: Optional["SummaryWriter"] = None,
):
    """
    Log metrics to the requested backends.
    """
    if not metrics:
        return

    if isinstance(backends, str):
        backends = [backends]

    formatted = _format_metrics(metrics, mode)

    for backend in backends:
        if backend == "wandb":
            if wandb.run is None:
                continue
            wandb.log(formatted, step=step)
        elif backend == "tensorboard":
            if writer is None:
                continue
            for key, value in formatted.items():
                writer.add_scalar(key, value, step)


def save_logs(log_data: dict, log_dir: str, filename: str = "logs.jsonl") -> None:
    """
    Persist metrics locally in JSONL format under the current run directory.
    """
    if not log_data:
        return
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    def default(o):
        if isinstance(o, float):
            return round(o, 6)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(log_path, "a") as f:
        f.write(json.dumps(log_data, default=default) + "\n")
