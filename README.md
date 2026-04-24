# C-MORAL

RL fine-tuning code for molecule optimization with `GRPO`, `GDPO`, and `PPO`.

This public version is prepared for GitHub upload:

- training outputs are ignored
- model weights are ignored
- test code is ignored
- local ADMET service files are ignored

## Recommended Environment

Recommended setup is a single conda environment named `molo`.

If you already have a working `molo` environment, you can continue using it directly.

### Option 1: create from `molo.yml`

```bash
conda env create -f molo.yml
conda activate molo
```

### Option 2: create a lighter environment manually

```bash
conda create -n molo python=3.11
conda activate molo
python -m pip install --upgrade pip
python -m pip install -r requirements-molo.txt
python -m pip install admet_ai==1.4.0
```

Notes:

- `train_grpo.py` and `train_gdpo.py` load `ADMETModel` directly in-process.
- You do not need to start `admet_server.py` for the current training path.
- Training assumes a CUDA-capable PyTorch environment.

## Repository Layout

- `train_rl.py`: unified launcher for GRPO and GDPO
- `train_grpo.py`: GRPO training entry
- `train_gdpo.py`: GDPO training entry
- `train_ppo.py`: PPO training entry
- `configs/`: experiment configs
- `rl/`: trainers, rollout buffer, PPO/GRPO/GDPO logic
- `utils/`: config loading, model loading, dataset loading
- `props/`: molecular property and reward helpers
- `data_new/`: local dataset cache or prepared dataset artifacts

## Before Training

You need local access to:

1. base model weights
2. LoRA initialization weights if your config uses them
3. dataset files, or network access for the first dataset/model download

The code will read local model folders first. If a configured model folder does not exist, it will try to download from Hugging Face and then cache it locally.

## How Config Paths Work

Configs in this repo still contain paths like `Molo/models/...`.

The loader normalizes these paths relative to the current repository root, so the code still works even if the GitHub repo name is `C-MORAL`.

## Training Workflow

### 1. Enter the repo

```bash
cd /path/to/C-MORAL
conda activate molo
```

### 2. Pick a config

Common examples:

- GRPO BPQ: `configs/train_config_mistral_bpq.yaml`
- GDPO BPQ: `configs/gdpo/train_config_mistral_bpq.yaml`
- default config: `configs/train_config.yaml`

## Models

This repo supports two base model families:

- `Mistral`
- `Llama`

Model choice is controlled by `model.base_model` in the config.

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

## Tasks

The repo includes prepared experiment aliases under `configs/exp/`:

- `abmp`: `amp+bbbp+mutag+plogp`
- `acep`
- `bcmq`
- `bdeq`
- `bdpq`
- `bpq`: `bbbp+plogp+qed`
- `cde`
- `dhmq`
- `elq`: `herg+liv+qed`
- `hlmpq`

Use an alias with:

```bash
python train_rl.py --algo grpo --exp bpq
python train_rl.py --algo gdpo --exp elq
```

If you want a specific model family, use the concrete config file instead of `--exp`.

## Default Configuration

The default config is [configs/train_config.yaml](/work/nvme/bfuy/rgao7/Molo/configs/train_config.yaml).

Its current defaults are:

- model: `Mistral`
- algorithm: `grpo`
- task: `carc+drd2+herg`
- dataset mode: `task`

So this command uses the default Mistral GRPO setup for `carc+drd2+herg`:

```bash
python train_rl.py --algo grpo --config configs/train_config.yaml
```

### 3. Start training

Run GRPO:

```bash
python train_rl.py --algo grpo --config configs/train_config_mistral_bpq.yaml
```

Run GDPO:

```bash
python train_rl.py --algo gdpo --config configs/gdpo/train_config_mistral_bpq.yaml
```

You can also launch by experiment alias:

```bash
python train_rl.py --algo grpo --exp bpq
python train_rl.py --algo gdpo --exp bpq
```

### 4. Optional: skip post-training test evaluation

```bash
python train_rl.py --algo grpo --config configs/train_config_mistral_bpq.yaml --skip_test_after_train
```

## Direct Entrypoints

If you do not want the wrapper:

```bash
python train_grpo.py --config configs/train_config_mistral_bpq.yaml
python train_gdpo.py --config configs/gdpo/train_config_mistral_bpq.yaml
python train_ppo.py --config configs/train_config.yaml
```

## Important Config Fields

In each config, check these sections first:

- `model.base_model`: `Mistral` or `Llama`
- `model.base_model_path`: local base model directory
- `model.lora_adapter_path`: local LoRA initialization directory
- `dataset.task_data_path` / `dataset.full_data_path`: dataset storage
- `rl_task.task`: property optimization target
- `logs.runs_dir`: run output directory
- `logs.logging_backend`: `wandb`, `tensorboard`, `both`, or `none`

## Example: BPQ Training

```bash
conda activate molo
python train_rl.py --algo grpo --exp bpq
```

This resolves to the BPQ GRPO config and writes outputs under the configured `runs_dir`.

## GitHub Upload Notes

The following are excluded by `.gitignore`:

- `models/`
- `runs/`
- `logs/`
- test code and test notebooks
- `admet_server.py`, `admet.yml`, `requirements-admet.txt`
- weight files such as `*.pt`, `*.pth`, `*.ckpt`, `*.bin`, `*.safetensors`

Check what will be committed with:

```bash
git status --short
```
