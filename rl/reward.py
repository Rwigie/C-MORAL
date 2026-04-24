import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen, rdFingerprintGenerator, DataStructs

from ..props import compute_chem_properties_batch, compute_chem_properties_remote, MOLO_PROPERTIES

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def compute_diff(source_props, target_props, keys):
    return {k: target_props.get(k, 0.0) - source_props.get(k, 0.0) for k in keys}

def compute_dynamic_weights(task_keys, mode="absolute", epsilon=1e-6):
    """
    Calculate normalized inverse weights for each property in task_keys based on their:
    1. Calculate original weight as 1/Target (the lower the target/required change, the higher the weight).
    2. Normalize all weights so that sum(weights) = 1.0.
    Args:
        task_keys (set): Set of property keys involved in the task (e.g., {'
        mode (str): 'absolute' or 'relative' weight calculation mode.
    Returns:
        dict: Normalized weights for each property key.
    """
    raw_weights = {}
    
    for k in task_keys:
        prop_key = k.lower()   
        prop_info = MOLO_PROPERTIES[prop_key]
        ref_val = 1.0
        if mode == "absolute":
            theta = prop_info.target_threshold
            if prop_info.optimization_direction == "minimize":
                ref_val = 1.0 - theta
            else:
                ref_val = theta 
        else: 
            ref_val = prop_info.delta_threshold
        raw_weights[prop_key] = 1.0 / max(ref_val, epsilon)
    total_raw_weight = sum(raw_weights.values())
    
    final_weights = {}
    if total_raw_weight > 0:
        for k, w in raw_weights.items():
            final_weights[k] = w / total_raw_weight
    else:
        return {}     
    return final_weights

def aggregate_rewards(rewards, reward_weight, keys, mode="absolute", aggregation="weighted_sum"):
    scores = {}
    for k in keys:
        val = rewards.get(k, 0.0)
        prop_key = k.lower()
        if mode == "absolute":
            if prop_key in MOLO_PROPERTIES:
                if MOLO_PROPERTIES[prop_key].optimization_direction == "minimize":
                    val = MOLO_PROPERTIES[prop_key].value_range[1] - val
        elif mode == "relative":
            if MOLO_PROPERTIES[prop_key].optimization_direction == "minimize":
                val = -val      
        scores[k] = val

    if aggregation == "weighted_sum":
        return sum(reward_weight.get(k, 0) * scores[k] for k in keys), scores
    if aggregation == "max":
        return max(scores.get(k, 0) for k in keys)  
    if aggregation == "min":
        return min(scores.get(k, 0) for k in keys)
    if aggregation == "mean":
        return sum(scores.get(k, 0) for k in keys) / len(keys)
    
    raise ValueError(
        f"Invalid aggregation method: {aggregation}, only 'weighted_sum', 'max', 'min', and 'mean' are supported."
    )

def compute_delta_penalty(diff, props_delta, props_delta_penalty, keys):
    penalty = 0.0
    penalty_per_prop = {k.lower(): 0.0 for k in keys}
    for k in keys:
        prop_key = k.lower()
        prop_info = MOLO_PROPERTIES[prop_key]
        delta_required = prop_info.delta_threshold     
        direction = prop_info.optimization_direction   
        actual_delta = diff.get(k, 0.0)
        failed = False
        if props_delta.get(k) is None:
            if direction == "maximize":
                if actual_delta < delta_required:
                    failed = True      
            elif direction == "minimize":
                if actual_delta > -delta_required:
                    failed = True
        else:
            if direction == "maximize":
                if actual_delta < props_delta[k]:
                    failed = True
            elif direction == "minimize":
                if actual_delta > -props_delta[k]:
                    failed = True
        if failed:
            penalty += props_delta_penalty.get(k, -3.0)
            penalty_per_prop[k] = props_delta_penalty.get(k, -3.0)
    return penalty, penalty_per_prop

def compute_range_penalty(props, props_range, props_range_penalty, keys):
    """Compute penalty for properties that fall outside the desired range.
    For each property, if its value does not meet the target threshold in the
    specified optimization direction, a penalty is applied.
    """
    penalty = 0.0
    penalty_per_prop = {k.lower(): 0.0 for k in keys}
    for k in keys:
        prop_key = k.lower()
        if prop_key not in MOLO_PROPERTIES:
            continue
        value = props.get(k)
        if value is None:
            continue
        prop_info = MOLO_PROPERTIES[prop_key]
        threshold = prop_info.target_threshold      
        direction = prop_info.optimization_direction 
        failed = False
        if props_range.get(k) is None:
            if direction == "maximize":
                if value < threshold:
                    failed = True    
            elif direction == "minimize":
                if value > threshold:
                    failed = True    

        else:
            if direction == "maximize":
                if value < props_range[k][0]:
                    failed = True
            elif direction == "minimize":
                if value > props_range[k][1]:
                    failed = True
        if failed:
            penalty_per_prop[k] = props_range_penalty.get(k, -3.0)
            penalty += props_range_penalty.get(k, -3.0)

    return penalty, penalty_per_prop

def compute_reward(
    source_smiles,
    target_smiles,
    task="qed+plogp",
    reward_mode="absolute",
    aggregation="weighted_sum",
    reward_weight=None,
    use_similarity=True,
    use_props_delta=True,
    props_delta=None,
    props_delta_penalty=None,
    use_props_range=True,
    props_range=None,
    props_range_penalty=None,
    scale=1.0,
    clip=1.0,
    admet_model=None,
):
    task_keys = set(task.lower().split("+"))
    if use_similarity:
        task_keys.add("sim")

    if isinstance(target_smiles[0], str):
        target_smiles = [[sm] for sm in target_smiles]

    if reward_weight is None:
        raise ValueError("reward_weight must be provided as a dictionary with keys like 'qed', 'plogp', 'sim'.")

    all_rewards = []
    valid_masks = []
    all_scores = []
    all_penalties = []
    diff_props_list = []
    target_props_list = []

    src_props, tgt_props = compute_chem_properties_remote(source_smiles, target_smiles, task=task)
    dynamic_reward_weight = compute_dynamic_weights(task_keys, mode=reward_mode)
    for src, tgt_list in zip(src_props, tgt_props):
        rewards = []
        scores = []
        penalties = []
        diff_props = []
        tgt_props_sublist = []

        for tgt in tgt_list:
            tgt_props_sublist.append(tgt)
            if not tgt["valid"]:
                rewards.append(-3.0)
                diff_props.append(None)
                valid_masks.append(0.0)
                scores.append({k: -1.0 for k in task_keys})
                continue
            valid_masks.append(1.0)
            diff = compute_diff(src, tgt, task_keys)
            if "sim" in task_keys:
                diff["sim"] = tgt.get("sim", 0.0)
            diff_props.append(diff)

            range_penalty, range_penalty_per_prop = (
                compute_range_penalty(tgt, props_range, props_range_penalty or {}, task_keys)
                if use_props_range
                else (0.0, {})
            )
            delta_penalty, delta_penalty_per_prop = (
                compute_delta_penalty(diff, props_delta, props_delta_penalty or {}, task_keys)
                if use_props_delta
                else (0.0, {})
            )
            penalty = range_penalty + delta_penalty
            
            if reward_mode == "absolute":
                reward, score = aggregate_rewards(tgt, dynamic_reward_weight, task_keys, reward_mode, aggregation=aggregation)
            elif reward_mode == "relative":
                reward, score = aggregate_rewards(diff, dynamic_reward_weight, task_keys, reward_mode, aggregation=aggregation)
            else:
                raise ValueError(f"Unsupported reward_mode: {reward_mode}")

            reward = reward * scale + penalty
            # reward = torch.tensor(reward).clamp(min=-clip, max=clip).item()
            rewards.append(reward)
            score = {k: scale * score.get(k, 0.0) + range_penalty_per_prop.get(k, 0.0) + delta_penalty_per_prop.get(k, 0.0) for k in task_keys }
            scores.append(score)
            penalties.append({'range_penalty': range_penalty, 'delta_penalty': delta_penalty})
        
        all_rewards.append(rewards)
        all_scores.append(scores)
        all_penalties.append(penalties)
        diff_props_list.append(diff_props)
        target_props_list.append(tgt_props_sublist)

    rewards_tensor = torch.tensor(all_rewards).flatten()
    valid_masks = torch.tensor(valid_masks, dtype=torch.float32).flatten()
    scores_dict = flatten_scores_dict(all_scores, task, use_similarity)
    return {
        "reward": rewards_tensor, # shape (B,)
        "scores": scores_dict,  # shape {key: Tensor(B,)}
        "valid_mask": valid_masks, # shape (B,)
        "penalties": all_penalties,
        "target_properties": target_props_list,
        "source_properties": src_props,
        "different_properties": diff_props_list,
    }

def flatten_scores_dict(scores_dict, task, use_similarity, device='cpu'):
    """
    Let List[List[Dict]] transform to Dict[str, Tensor(B)]
    
    Args:
        raw_batch_rewards: nested list of dicts. 
                           Shape: [num_envs, group_size, dict_keys]
    Returns:
        scores_dict: {key: Tensor(Batch_Size,)} where Batch_Size = num_envs * group_size
    """
    keys = task.split('+')
    if use_similarity:
        keys.append('sim')
    parsed_data = {k: [] for k in keys}
    for group in scores_dict:       
        for sample_metrics in group:
            if sample_metrics is None:
                for k in keys:
                    parsed_data[k].append(-1.0)  
            else:
                for k, v in sample_metrics.items():
                    parsed_data[k].append(v)
    tensor_scores = {
        k: torch.tensor(v, dtype=torch.float32, device=device) 
        for k, v in parsed_data.items()
    }

    return tensor_scores

if __name__ == "__main__":
    # For quick testing
    source = ["C"]  # methane
    target = [["CO"]]  # methanol
    reward_weight = {"qed": 1.0, "plogp": 0.0, "sim": 0.0, 'hia': 0.0}
    result = compute_reward(
        source_smiles=source,
        target_smiles=target,
        task="qed+plogp+hia",
        reward_weight=reward_weight,
        use_similarity=False,
        use_props_delta=False,
        use_props_range=False,
    )
    print("Result:", result)
