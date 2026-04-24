import math

import torch


def _infer_total_samples(result: dict) -> int:
    target_smiles = result.get("target_smiles")
    if isinstance(target_smiles, list):
        if target_smiles and isinstance(target_smiles[0], list):
            return sum(len(x) for x in target_smiles if isinstance(x, list))
        return len(target_smiles)

    rewards = result.get("rewards")
    if isinstance(rewards, torch.Tensor):
        return rewards.numel()
    if isinstance(rewards, list):
        if rewards and isinstance(rewards[0], list):
            return sum(len(x) for x in rewards if isinstance(x, list))
        return len(rewards)

    scores = result.get("scores")
    if isinstance(scores, dict):
        for v in scores.values():
            if isinstance(v, torch.Tensor):
                return v.numel()
            try:
                v_list = list(v)
            except TypeError:
                continue
            return len(v_list)

    return 0

def compute_property_mean(result: dict, task: str) -> dict:
    """
    Compute per-property mean values for generated (target) molecules, based on `task` keys.

    Notes:
    - `result["target_properties"]` may be `List[List[dict]]` or already `List[dict]`.
    - Averages are computed over the total number of generated targets (i.e. `len(target_smiles)`),
      not over the number of valid targets.
    - Invalid targets contribute 0 to the mean (no invalid-related metrics are returned).
    - Always includes `sim` mean in addition to `task` keys.
    """
    task_keys = [k.strip().lower() for k in (task or "").split("+") if k.strip()]
    keys = list(dict.fromkeys([*task_keys, "sim"]))
    target_properties_raw = result.get("target_properties", [])
    target_properties = []
    for item in target_properties_raw:
        if isinstance(item, dict):
            target_properties.append(item)
        elif isinstance(item, list):
            target_properties.extend([d for d in item if isinstance(d, dict)])

    denom = _infer_total_samples(result) or len(target_properties)
    if denom == 0:
        return {}

    prop_sums = {k: 0.0 for k in keys}
    for prop in target_properties:
        if prop.get("valid") is False:
            continue
        for k in keys:
            v = prop.get(k)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            prop_sums[k] += v

    return {f'{k}_value': (prop_sums[k] / denom) for k in keys}

def compute_score_mean(result: dict, task: str) -> dict:
    """
    Compute per-property mean values for score tensors or score dicts.

    Notes:
    - `result["scores"]` can be a dict of tensors (preferred) or a list of dicts.
    - Averages are computed over the total number of scores.
    - Always includes `sim` mean in addition to `task` keys.
    """
    task_keys = [k.strip().lower() for k in (task or "").split("+") if k.strip()]
    keys = list(dict.fromkeys([*task_keys, "sim"]))
    scores = result.get("scores")
    if scores is None:
        return {}

    if isinstance(scores, dict):
        means = {}
        for k in keys:
            v = scores.get(k)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if v.numel() == 0:
                    continue
                denom = _infer_total_samples(result) or v.numel()
                mean_val = (v.float().sum().item() / denom)
            else:
                try:
                    v_list = list(v)
                except TypeError:
                    continue
                if not v_list:
                    continue
                denom = _infer_total_samples(result) or len(v_list)
                mean_val = float(sum(v_list)) / denom
            if not math.isfinite(mean_val):
                continue
            means[f"{k}_score"] = mean_val
        return means

    if isinstance(scores, list):
        flat_scores = []
        for item in scores:
            if isinstance(item, dict):
                flat_scores.append(item)
            elif isinstance(item, list):
                flat_scores.extend([d for d in item if isinstance(d, dict)])
        denom = _infer_total_samples(result) or len(flat_scores)
        if denom == 0:
            return {}
        scores_sum = {k: 0.0 for k in keys}
        for score in flat_scores:
            for k in keys:
                v = score.get(k)
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    v = v.item()
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(v):
                    continue
                scores_sum[k] += v
        return {f"{k}_score": (scores_sum[k] / denom) for k in keys}

    return {}


def check_lora_weights(model, step_name="", num_params=4, verbose=True, include_frozen=True):
    """
    Check and print statistics of LoRA weights in the given model.
    
    Args:
        model: The model (Actor, Critic, or nn.Module) containing LoRA parameters.
        step_name: Optional name for the current step (for logging purposes).
        num_params: Number of LoRA parameters to display.
        verbose: Whether to print the statistics.
    Returns:
        dict: Statistics of all LoRA parameters.
    """
    # If an Actor/Critic object is passed, get its model attribute
    if hasattr(model, 'model'):
        model = model.model
    
    lora_params_info = {}
    count = 0
    
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and (param.requires_grad or include_frozen):
            lora_params_info[name] = {
                'shape': param.shape,
                'mean': param.data.float().mean().item(),
                'std': param.data.float().std().item(),
                'min': param.data.float().min().item(),
                'max': param.data.float().max().item(),
                'first_10': param.data.flatten()[:10].cpu().tolist()
            }
            count += 1
            if count >= num_params:
                break
    
    if verbose:
        header = f"🔍 LORA WEIGHTS CHECK: {step_name}" if step_name else "🔍 LORA WEIGHTS CHECK"
        print("\n" + "="*80)
        print(header)
        print("="*80)
        
        if not lora_params_info:
            print("⚠️ No LoRA parameters found!")
        else:
            print(f"\n📊 LoRA Weights Statistics (first {len(lora_params_info)} params):\n")
            for name, info in lora_params_info.items():
                print(f"{name}:")
                print(f"  Shape: {info['shape']}")
                print(f"  Mean: {info['mean']:.6f}, Std: {info['std']:.6f}")
                print(f"  Range: [{info['min']:.6f}, {info['max']:.6f}]")
                print(f"  First 10 values: {[f'{v:.4f}' for v in info['first_10']]}")
                
                # Warn if std is very small
                if info['std'] < 1e-6:
                    print(f"  ⚠️ WARNING: All values are near zero (std={info['std']:.8f})")
                print()
        
        print("="*80 + "\n")
    
    return lora_params_info
