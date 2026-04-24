# Experiment Configs for `train_rl.py`

`train_rl.py --exp <name>` supports two formats:

1) Base + overrides (recommended)

Fields:
- `name`
- `grpo_base`
- `gdpo_base`
- `common_overrides`
- `grpo_overrides`
- `gdpo_overrides`

At runtime, `train_rl.py` deep-merges:
- `grpo`: `grpo_base + common_overrides + grpo_overrides`
- `gdpo`: `gdpo_base + common_overrides + gdpo_overrides`

2) Legacy direct mapping (still supported)

Fields:
- `grpo_config`
- `gdpo_config`

Examples:
- `python -m Molo.train_rl --algo grpo --exp elq`
- `python -m Molo.train_rl --algo gdpo --exp elq`
