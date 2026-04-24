import re
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import PreTrainedTokenizer


def extract_smiles(text: str) -> str:
    match = re.search(r"<SMILES>\s*(.*?)\s*</SMILES>", text)
    return match.group(1) if match else None


def generate_smiles(
    prompt: List[str],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    num_envs: int = 1,
    num_return_sequences: int = 10,
    num_beams: int = 1,
    do_sample: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]], Dict[str, torch.Tensor]]:
    model.eval()
    num_envs = len(prompt)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    prompt_tensor_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_instr = inputs

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        num_beams=num_beams,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        renormalize_logits=True,
        use_cache=True,
        output_scores=False,
    )

    inputs_ids = outputs
    pad_id = tokenizer.pad_token_id
    attention_mask = (inputs_ids != pad_id).long()
    action_mask = attention_mask.clone()

    action_mask[:, :prompt_tensor_len] = 0
    labels = inputs_ids.clone()
    labels[action_mask == 0] = -100

    target_smiles = []
    raw_list = []
    reshaped_outputs = inputs_ids.view(num_envs, num_return_sequences, -1)
    reshaped_masks = action_mask.view(num_envs, num_return_sequences, -1)

    for i in range(num_envs):
        target_smile = []
        for j in range(num_return_sequences):
            valid_tokens = reshaped_outputs[i, j][reshaped_masks[i, j] == 1]
            result = tokenizer.decode(valid_tokens, skip_special_tokens=True)
            target_smile.append(extract_smiles(result))
        raw_list.append(result)
        target_smiles.append(target_smile)
    
    return inputs_ids, attention_mask, labels, target_smiles, prompt_instr


def generate_smiles_eval(
    prompt_instr,
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    num_envs: int = 1,
    num_return_sequences: int = 10,
    num_beams: int = 1,
    do_sample: bool = False,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    device: str = "cpu",
):
    inputs = prompt_instr
    attention_mask = inputs.attention_mask
    inputs_len = attention_mask.sum(dim=1)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        renormalize_logits=True,
    )

    attention_mask_full = torch.ones_like(outputs)
    target_smiles = []
    for i in range(num_envs):
        env_smiles = []
        for j in range(num_return_sequences):
            idx = i * num_return_sequences + j
            valid_length = attention_mask_full[idx].sum().item()
            generated_tokens = outputs[idx][inputs_len[i] : valid_length].cpu()
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            env_smiles.append(extract_smiles(result))
        target_smiles.append(env_smiles)
    
    return target_smiles

