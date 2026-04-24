import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def is_valid_model_dir(path: str) -> bool:
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        return False
    for filename in os.listdir(path):
        if filename.endswith(".bin") or filename.endswith(".safetensors"):
            return True
    return False


def load_model_tokenizer(
    base_model: str = "Mistral",
    base_model_path: str = "models/Mistral/base",
    lora_adapter_path: str = "models/Mistral/gellmoC-lora",
    use_lora: bool = True,
    device: str = "cpu",
):
    if base_model not in {"Mistral", "Llama"}:
        raise ValueError("base_model must be 'Mistral' or 'Llama'")

    if base_model == "Mistral":
        base_model_hf_path = "mistralai/Mistral-7B-Instruct-v0.3"
        lora_adapter_hf_path = "NingLab/GeLLMO-C-P10-Mistral"
    elif base_model == "Llama":
        base_model_hf_path = "meta-llama/Llama-3.1-8B-Instruct"
        lora_adapter_hf_path = "NingLab/GeLLMO-C-P10-Llama"

    if os.path.exists(base_model_path):
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_hf_path)
        tokenizer.save_pretrained(base_model_path)

    # Keep train/test behavior consistent with model-specific padding behavior.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if base_model == "Llama":
            # Llama-3 tokenizer ships with this reserved right-pad token.
            llama_pad = "<|finetune_right_pad_id|>"
            vocab = tokenizer.get_vocab()
            if llama_pad in vocab:
                tokenizer.pad_token = llama_pad
            elif tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
        else:  # Mistral
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            elif tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32

    if is_valid_model_dir(base_model_path):
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch_dtype,
            device_map=None,
            local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_hf_path,
            dtype=torch_dtype,
            device_map=None,
        )
        model.save_pretrained(base_model_path)

    if use_lora:
        if os.path.exists(lora_adapter_path):
            model = PeftModel.from_pretrained(
                model,
                lora_adapter_path,
                local_files_only=True,
                is_trainable=True,
            )
        else:
            model = PeftModel.from_pretrained(
                model,
                lora_adapter_hf_path,
                is_trainable=True,
            )
            model.save_pretrained(lora_adapter_path)

        for name, param in model.named_parameters():
            if "lora_" not in name.lower():
                param.requires_grad = False

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model = model.to(device)
    return model, tokenizer
