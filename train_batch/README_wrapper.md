# Unified RL Launch Wrapper (Non-invasive)

This wrapper does **not** modify any existing training code or existing sbatch files.

## 1) Local dry-run (check command only)

```bash
bash scripts/run_rl.sh --algo gdpo --exp elq --dry-run
```

## 2) Local run

```bash
bash scripts/run_rl.sh --algo grpo --task elq
```

## 3) sbatch run (single wrapper)

```bash
sbatch --export=ALGO=gdpo,TASK=elq train_batch/train_rl_wrapper.slurm
```

Optional extra args:

```bash
sbatch --export=ALGO=gdpo,TASK=elq,EXTRA_ARGS="--skip_test_after_train" train_batch/train_rl_wrapper.slurm
```

Or use experiment alias:

```bash
sbatch --export=ALGO=gdpo,EXP=elq train_batch/train_rl_wrapper.slurm
```

## Config mapping

- `grpo`: `configs/train_config_mistral_<task>.yaml`
- `gdpo`: `configs/gdpo/train_config_mistral_<task>.yaml`
- `exp` alias: `configs/exp/<exp>.yaml` (mapped by `train_rl.py`)

Example task names in current repo: `acep`, `abmp`, `bcmq`, `bdeq`, `bdpq`, `bpq`, `cde`, `dhmq`, `elq`, `hlmpq`.
