# C-MORAL: Controllable Multi-Objective Molecular Optimization with Reinforcement Alignment for LLMs

<p align="center">
  <a href="https://github.com/Rwigie/C-MORAL"><img src="https://img.shields.io/badge/GitHub-C--MORAL-181717?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/Rwigle/C-MORAL-Mistral-GRPO"><img src="https://img.shields.io/badge/HuggingFace-Mistral--GRPO-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="https://huggingface.co/Rwigle/C-MORAL-Mistral-GDPO"><img src="https://img.shields.io/badge/HuggingFace-Mistral--GDPO-FFB000?logo=huggingface&logoColor=black" alt="Hugging Face GDPO"></a>
  <a href="https://huggingface.co/datasets/NingLab/C-MuMOInstruct"><img src="https://img.shields.io/badge/Dataset-C--MuMOInstruct-8A2BE2?logo=huggingface&logoColor=white" alt="Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Models-Mistral%20%7C%20Llama-4A5568" alt="Models">
</p>

Official codebase for **C-MORAL**, a controllable reinforcement alignment framework for multi-objective molecular optimization with large language models.

This repository contains training pipelines for:

- `GRPO`
- `GDPO`
- `PPO`

with support for both:

- `Mistral`
- `Llama`

## Overview

C-MORAL aligns molecular generation with multiple controllable property objectives by combining instruction-based generation with reinforcement learning over molecular reward signals.

The codebase includes:

- multi-task molecular optimization training
- `Mistral` and `Llama` LoRA fine-tuning
- `GRPO`, `GDPO`, and `PPO` training pipelines
- reusable config-driven experiment management
- local adapter loading and Hugging Face-compatible model export

## Framework

<p align="center">
  <img src="./figs/framework.png" alt="C-MORAL framework" width="900">
</p>

## Highlights

- Unified RL training entrypoint via `train_rl.py`
- Config-based experiment control for tasks, models, and algorithms
- Multi-objective task suites such as `bpq`, `elq`, `abmp`, and `hlmpq`
- Hugging Face-ready LoRA adapters for public release
- Repository prepared for public sharing without weights, runtime logs, or private local artifacts

## Model Zoo

Current Hugging Face model repositories:

- [Mistral GRPO Adapters](https://huggingface.co/Rwigle/C-MORAL-Mistral-GRPO)
- [Mistral GDPO Adapters](https://huggingface.co/Rwigle/C-MORAL-Mistral-GDPO)

Recommended subfolders in the Hugging Face repository follow task aliases:

- `abmp`
- `acep`
- `bcmq`
- `bdeq`
- `bdpq`
- `bpq`
- `cde`
- `dhmq`
- `elq`
- `hlmpq`

Example PEFT loading pattern:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_repo = "Rwigle/C-MORAL-Mistral-GRPO"
task_subfolder = "bpq"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(model, adapter_repo, subfolder=task_subfolder)
```

## Dataset

Training and evaluation rely on the `C-MuMOInstruct` dataset:

- [NingLab/C-MuMOInstruct](https://huggingface.co/datasets/NingLab/C-MuMOInstruct)

If the configured local dataset cache does not exist, the code will fetch the dataset from Hugging Face and cache it locally.

## Installation

Recommended setup is a single conda environment named `molo`.

### Option 1: create from `molo.yml`

```bash
conda env create -f molo.yml
conda activate molo
```

### Option 2: create manually

```bash
conda create -n molo python=3.11
conda activate molo
python -m pip install --upgrade pip
python -m pip install -r requirements-molo.txt
python -m pip install admet_ai==1.4.0
```

Notes:

- `train_grpo.py` and `train_gdpo.py` load `ADMETModel` directly in-process.
- The current public training path does not require `admet_server.py`.
- Training assumes a CUDA-capable PyTorch environment.

## Repository Structure

- `train_rl.py`: unified launcher for `GRPO` and `GDPO`
- `train_grpo.py`: GRPO entrypoint
- `train_gdpo.py`: GDPO entrypoint
- `train_ppo.py`: PPO entrypoint
- `configs/`: experiment and model configs
- `rl/`: RL trainers, rollout buffer, and objective code
- `utils/`: config loading, model loading, dataset loading
- `props/`: molecular property and reward utilities
- `data_new/`: local dataset cache or prepared dataset artifacts

## Supported Models

This repository supports two base model families:

- `Mistral`
- `Llama`

Typical config naming:

- `configs/train_config_mistral_*.yaml`: Mistral + GRPO
- `configs/gdpo/train_config_mistral_*.yaml`: Mistral + GDPO
- `configs/train_config_llama_*.yaml`: Llama + GRPO
- `configs/gdpo/train_config_llama_*.yaml`: Llama + GDPO

Examples:

```bash
python train_rl.py --algo grpo --config configs/train_config_mistral_bpq.yaml
python train_rl.py --algo grpo --config configs/train_config_llama_bpq.yaml
python train_rl.py --algo gdpo --config configs/gdpo/train_config_mistral_bpq.yaml
python train_rl.py --algo gdpo --config configs/gdpo/train_config_llama_bpq.yaml
```

## Supported Tasks

Prepared task aliases under `configs/exp/`:

- `abmp`: `amp+bbbp+mutag+plogp`
- `acep`: `amp+carc+herg+plogp`
- `bcmq`: `bbbp+carc+mutag+qed`
- `bdeq`: `bbbp+drd2+herg+qed`
- `bdpq`: `bbbp+drd2+qed+plogp`
- `bpq`: `bbbp+plogp+qed`
- `cde`: `carc+drd2+herg`
- `dhmq`: `drd2+hia+mutag+qed`
- `elq`: `herg+liv+qed`
- `hlmpq`: `hia+liv+mutag+plogp+qed`

Use an alias with:

```bash
python train_rl.py --algo grpo --exp bpq
python train_rl.py --algo gdpo --exp elq
```

If you want explicit model control, use a concrete config path instead of `--exp`.

## Default Setup

The default config is [configs/train_config.yaml](/work/nvme/bfuy/rgao7/Molo/configs/train_config.yaml).

Its current defaults are:

- model: `Mistral`
- algorithm: `grpo`
- task: `carc+drd2+herg`
- dataset mode: `task`

Run the default configuration with:

```bash
python train_rl.py --algo grpo --config configs/train_config.yaml
```

## Training

### GRPO

```bash
python train_rl.py --algo grpo --config configs/train_config_mistral_bpq.yaml
```

### GDPO

```bash
python train_rl.py --algo gdpo --config configs/gdpo/train_config_mistral_bpq.yaml
```

### Launch by task alias

```bash
python train_rl.py --algo grpo --exp bpq
python train_rl.py --algo gdpo --exp bpq
```

### Skip post-training test evaluation

```bash
python train_rl.py --algo grpo --config configs/train_config_mistral_bpq.yaml --skip_test_after_train
```

## Direct Entrypoints

```bash
python train_grpo.py --config configs/train_config_mistral_bpq.yaml
python train_gdpo.py --config configs/gdpo/train_config_mistral_bpq.yaml
python train_ppo.py --config configs/train_config.yaml
```

## Important Config Fields

- `model.base_model`: `Mistral` or `Llama`
- `model.base_model_path`: local base model directory
- `model.lora_adapter_path`: local LoRA initialization directory
- `dataset.task_data_path` / `dataset.full_data_path`: dataset storage
- `rl_task.task`: property optimization target
- `logs.runs_dir`: run output directory
- `logs.logging_backend`: `wandb`, `tensorboard`, `both`, or `none`

## Path Resolution

Some configs still contain paths like `Molo/models/...`.

The loader normalizes these paths relative to the current repository root, so training still works even if the public repository name is `C-MORAL`.

## Public Release Notes

The public repository excludes:

- model weights
- runtime outputs under `runs/` and `logs/`
- local test code
- local-only ADMET service files

Check the public commit contents with:

```bash
git status --short
```

## Citation

If you use this repository, please cite:

```text
C-MORAL: Controllable Multi-Objective Molecular Optimization with
Reinforcement Alignment for LLMs
```
