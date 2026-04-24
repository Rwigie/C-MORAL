import os
import torch


def save_checkpoint(
    actor,
    optimizer_actor,
    critic,
    optimizer_critic,
    save_dir,
    iteration: int = None,
    save_full_critic: bool = False,
):
    """
    Save actor/critic parameters and optimizer states under the run's checkpoint directory.
    """
    target_dir = save_dir if iteration is None else os.path.join(save_dir, f"iter_{iteration}")
    os.makedirs(target_dir, exist_ok=True)

    actor.save_lora(target_dir)
    torch.save(optimizer_actor.state_dict(), os.path.join(target_dir, "optimizer_actor.pt"))

    critic.save_value_head(target_dir)
    if save_full_critic:
        torch.save(critic.state_dict(), os.path.join(target_dir, "critic.pt"))
    torch.save(optimizer_critic.state_dict(), os.path.join(target_dir, "optimizer_critic.pt"))


def save_checkpoint_actor(
    actor,
    optimizer_actor,
    save_dir,
    iteration: int = None,
):
    """
    Save actor parameters and optimizer states under the run's checkpoint directory.
    """
    target_dir = save_dir if iteration is None else os.path.join(save_dir, f"iter_{iteration}")
    os.makedirs(target_dir, exist_ok=True)

    actor.save_lora(target_dir)
    torch.save(optimizer_actor.state_dict(), os.path.join(target_dir, "optimizer_actor.pt")
)

def save_checkpoint_critic(
    critic,
    optimizer_critic,
    save_dir,
    iteration: int = None,
    save_full_critic: bool = False,
):
    """
    Save critic parameters and optimizer states under the run's checkpoint directory.
    """
    target_dir = save_dir if iteration is None else os.path.join(save_dir, f"iter_{iteration}")
    os.makedirs(target_dir, exist_ok=True)

    critic.save_value_head(target_dir)
    if save_full_critic:
        torch.save(critic.state_dict(), os.path.join(target_dir, "critic.pt"))
    torch.save(optimizer_critic.state_dict(), os.path.join(target_dir, "optimizer_critic.pt"))