import os

import torch
import torch.nn as nn


class MoloCritic(nn.Module):
    """
    Critic head sitting on top of the language model backbone.
    """

    def __init__(self, model, device: str = "cpu"):
        super().__init__()
        self.backbone = model
        self.device = device
        head_dtype = next(self.backbone.parameters()).dtype
        self.value_head = nn.Linear(self.backbone.config.hidden_size, 1)
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)
        self.value_head.to(device=self.device, dtype=torch.float32)

    def save_value_head(self, save_dir) -> None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "value_head.pt")
        torch.save(self.value_head.state_dict(), path)

    def load_value_head(self, load_dir) -> None:
        path = os.path.join(load_dir, "value_head.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Value head file not found at {path}")
        self.value_head.load_state_dict(torch.load(path, map_location=self.device))

    def set_gradient_checkpointing(self, enable: bool) -> None:
        fn_name = "gradient_checkpointing_enable" if enable else "gradient_checkpointing_disable"
        backbone = self.backbone
        if hasattr(backbone, fn_name):
            getattr(backbone, fn_name)()
        elif hasattr(backbone, "base_model") and hasattr(backbone.base_model, fn_name):
            getattr(backbone.base_model, fn_name)()

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1].to(torch.float32)
        values = self.value_head(last_hidden).squeeze(-1)
        return values
