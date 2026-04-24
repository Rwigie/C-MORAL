from typing import Tuple, List

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedTokenizer

from .utils.generation import generate_smiles, generate_smiles_eval


class MoloActor(nn.Module):
    """
    Thin wrapper around the language model used as PPO actor.
    Provides helper methods for generation and adapter saving.
    """

    def __init__(self, model: PeftModel, tokenizer: PreTrainedTokenizer, device: str = "cpu"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompt,
        num_return_sequences: int = 10,
        num_envs: int = 1,
        num_beams: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 100,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], dict]:
        self.model.eval()
        inputs_ids, attention_mask, labels, target_smiles, prompt_instr = generate_smiles(
            prompt=prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            num_envs=num_envs,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            device=self.device,
        )
        return inputs_ids, attention_mask, labels, target_smiles, prompt_instr

    def generate_eval(
        self,
        prompt_instr: torch.Tensor,
        num_envs: int = 1,
        num_return_sequences: int = 10,
        num_beams: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 100,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ):
        self.model.eval()
        target_smiles = generate_smiles_eval(
            prompt_instr=prompt_instr,
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            num_envs=num_envs,
        )
        return target_smiles

    def save_lora(self, save_dir) -> None:
        import os

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "lora_adapter")
        self.model.save_pretrained(path, save_embedding_layers=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits
    
