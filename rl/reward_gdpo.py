import torch
from ..props import compute_chem_properties_batch, MOLO_PROPERTIES

# --- 1. 核心打分逻辑 (业务层: Split Reward Strategy) ---




def get_raw_molo_score(current_val, source_val, prop_key):
    # 1. 基础检查
    if prop_key not in MOLO_PROPERTIES:
        return current_val
    
    prop_info = MOLO_PROPERTIES[prop_key]
    target = prop_info.target_threshold
    direction = prop_info.optimization_direction
    
    # === 关键修正开始 ===
    # 不再使用简单的取反，而是使用 (Max - Val) 进行映射
    # 这保证了 Minimize 任务的优等生也能拿到巨大的正分
    
    if direction == "maximize":
        # Maximize (QED): 越大越好，直接用
        curr_score = current_val
        src_score = source_val
        target_score = target
    else:
        # Minimize (SA Score): 越小越好
        # 使用上界进行翻转: Score = Max_Bound - Val
        # 假设 SA 范围 [1, 10]，那么 1->9(优), 10->0(差)
        
        # 必须确保 prop_info 里有 value_range，如果没有，给个默认大值
        max_bound = prop_info.value_range[1] 
        
        curr_score = max_bound - current_val
        src_score = max_bound - source_val
        target_score = max_bound - target
    # === 关键修正结束 ===

    # 下面的逻辑保持完全不变 (Split Strategy)
    
    # 1. 判断是否达标 (注意: 因为已经翻转了数值，这里永远是 >= target_score)
    is_optimal = (curr_score >= target_score)
    src_is_optimal = (src_score >= target_score)

    # Case A: Source 已经是优等生 -> 维稳/保持 (返回绝对值)
    if src_is_optimal:
        return curr_score

    # Case B: Source 是差生
    else:
        # 差生变成了优等生 -> 奖励登顶 (返回绝对值)
        if is_optimal:
            return curr_score
        
        # 差生还是差生 -> 鼓励进步 (返回 Delta)
        else:
            delta = curr_score - src_score
            return delta


def compute_sigmoid_score(val, target, k=10.0, direction="maximize"):
    if val is None:
        return 1e-6
    x = torch.tensor(float(val), dtype=torch.float32)
    t = torch.tensor(float(target), dtype=torch.float32)
    if direction == "maximize":
        return torch.sigmoid(k * (x - t)).item()
    if direction == "minimize":
        return torch.sigmoid(k * (t - x)).item()
    return 0.0


def _resolve_target_threshold(prop_key, direction, target_thresholds=None, props_range=None, use_props_range=True):
    # 1) per-sample dynamic target (highest priority)
    if isinstance(target_thresholds, dict) and prop_key in target_thresholds:
        return float(target_thresholds[prop_key]), "sample_target"

    # 2) static threshold from property metadata
    if prop_key in MOLO_PROPERTIES:
        return float(MOLO_PROPERTIES[prop_key].target_threshold), "static_theta"

    # 3) config range target (lowest priority)
    if use_props_range and isinstance(props_range, dict) and prop_key in props_range:
        rng = props_range.get(prop_key)
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
            return (lo if direction == "maximize" else hi), "range_target"

    # fallback
    return None, "none"
        


# --- 2. GDPO 主流程 (算法层) ---

def compute_reward_gdpo(
    source_smiles,
    target_smiles,
    task="qed+plogp",
    reward_weight=None,
    use_similarity=True,
    invalid_penalty=-2.0, # 给 Invalid 分子的温和负分 (相对于 Z-score)
    scale=1.0,
    admet_model=None,
    # 以下参数保留用于接口兼容，但在 GDPO 逻辑中不再直接使用
    reward_mode="absolute",
    aggregation="weighted_sum",
    use_props_delta=True,
    props_delta=None,
    props_delta_penalty=None,
    use_props_range=True,
    props_range=None,
    props_range_penalty=None,
    clip=1.0,
    sigmoid_k_default=10.0,
    target_thresholds_batch=None,
):
    """
    GDPO 风格的 Reward 计算函数。
    使用流程: Raw score -> Sigmoid -> Group-wise Normalization -> Weighted Sum。
    """
    
    # 1. 解析任务 Key
    task_keys = set(task.lower().split("+"))
    if use_similarity:
        task_keys.add("sim")

    # 权重默认处理
    if reward_weight is None:
        reward_weight = {k: 1.0 for k in task_keys}

    # 2. 预处理输入
    if isinstance(target_smiles[0], str):
        target_smiles = [[sm] for sm in target_smiles]

    # 3. 计算属性 (复用原有的 batch 计算函数)
    # src_props_list: List[Dict]
    # tgt_props_list: List[List[Dict]]
    src_props_list, tgt_props_list = compute_chem_properties_batch(source_smiles, target_smiles, task=task, admet_model=admet_model)

    # 4. 结果容器
    all_final_advantages = [] 
    all_log_scores = []   # 用于 logging，存 Sigmoid 后的 score
    valid_masks = []
    
    # 用于日志记录 (直接透传)
    tgt_prop_logs = [] 

    # 5. GDPO 核心循环: 遍历每个 Prompt (Group)
    for sample_idx, (src_p, tgt_group) in enumerate(zip(src_props_list, tgt_props_list)):
        sample_target_thresholds = None
        if isinstance(target_thresholds_batch, list) and sample_idx < len(target_thresholds_batch):
            sample_target_thresholds = target_thresholds_batch[sample_idx]
        
        tgt_prop_logs.append(tgt_group) # Log
        
        # 5.1 收集当前 Group 的 Raw Rewards 和 Validity
        # 结构: {'qed': [r1, r2...], 'plogp': [r1, r2...]}
        group_raw_rewards = {k: [] for k in task_keys}
        group_valid_flags = []
        group_phys_values = [] # 保留真实物性，供 target_properties 统计 value
        group_sigmoid_scores = [] # 用于 scores 输出（Sigmoid 后）
        
        for tgt_p in tgt_group:
            is_valid = tgt_p.get("valid", False)
            group_valid_flags.append(1.0 if is_valid else 0.0)

            phys_vals = {}
            for k in task_keys:
                if is_valid:
                    phys_vals[k] = tgt_p.get(k, 0.0)
                else:
                    if k in MOLO_PROPERTIES:
                        prop_info = MOLO_PROPERTIES[k]
                        direction = prop_info.optimization_direction
                        if direction == "minimize":
                            phys_vals[k] = prop_info.value_range[1]
                        else:
                            phys_vals[k] = prop_info.value_range[0]
                    else:
                        phys_vals[k] = 0.0
            group_phys_values.append(phys_vals)

            sample_sigmoid_scores = {}
            for k in task_keys:
                if not is_valid:
                    group_raw_rewards[k].append(None)
                    sample_sigmoid_scores[k] = 0.0
                
                else:
                    val_new = tgt_p.get(k, 0.0)
                    if k == "sim":
                        score = compute_sigmoid_score(
                            val=val_new,
                            target=0.6,
                            k=60.0,
                            direction="maximize",
                        )
                    elif k in MOLO_PROPERTIES:
                        prop_info = MOLO_PROPERTIES[k]
                        direction = prop_info.optimization_direction
                        delta = float(getattr(prop_info, "delta_threshold", 0.0) or 0.0)
                        current_k = (4.0 / delta) if delta > 0 else sigmoid_k_default
                        target, _ = _resolve_target_threshold(
                            prop_key=k,
                            direction=direction,
                            target_thresholds=sample_target_thresholds,
                            props_range=props_range,
                            use_props_range=use_props_range,
                        )
                        score = compute_sigmoid_score(
                            val=val_new,
                            target=target if target is not None else 0.0,
                            k=current_k,
                            direction=direction,
                        )
                    else:
                        score = 0.0
                    group_raw_rewards[k].append(score)
                    sample_sigmoid_scores[k] = score
            group_sigmoid_scores.append(sample_sigmoid_scores)

        # GDPO Step 1: Sigmoid + Decoupled Normalization (组内分别归一化)
        normalized_advantages = {}
        epsilon = 1e-8

        for k in task_keys:
            valid_vals = [v for v in group_raw_rewards[k] if v is not None]
        
            if len(valid_vals) > 0:
                min_valid = min(valid_vals)
                invalid_fill_val = min_valid - abs(min_valid * 0.5)
            else:
                invalid_fill_val = -5.0 

            full_vals = [v if v is not None else invalid_fill_val for v in group_raw_rewards[k]]
            vals_tensor = torch.tensor(full_vals, dtype=torch.float32)
            if len(vals_tensor) > 1 and vals_tensor.std() > 0:
                mean = vals_tensor.mean()
                std = vals_tensor.std()
                z_scores = (vals_tensor - mean) / (std + epsilon)
            else:
                z_scores = vals_tensor - vals_tensor.mean()
            
            normalized_advantages[k] = z_scores

        # 5.3 GDPO Step 2: Weighted Sum (加权求和)
        # 论文核心: A_sum = w1*A1 + w2*A2 ...
        # # ----------- Linear Mean -----------
        group_total_advantage = torch.zeros(len(tgt_group))
        for k in task_keys:
            w = reward_weight.get(k, 1.0)
            group_total_advantage += normalized_advantages[k]
        
        # adv_list = []
        # for k in task_keys:
        #     w = reward_weight.get(k, 1.0)
        #     adv_list.append(normalized_advantages[k])

        # # 2. 堆叠成 Tensor [Batch, Group, Num_Tasks]
        # adv_stack = torch.stack(adv_list, dim=-1)

        # # 3. 一次性计算 Soft-Min
        # # 公式: -T * log( sum( exp( -x / T ) ) )
        # T = 0.8
        # N = adv_stack.size(-1)
        # group_total_advantage = -T * torch.logsumexp(-adv_stack / T, dim=-1) + T * torch.log(torch.tensor(float(N), device=adv_stack.device))


        # 保存结果
        all_final_advantages.append(group_total_advantage)
        valid_masks.append(torch.tensor(group_valid_flags))
        all_log_scores.append(group_sigmoid_scores)

    # 6. 数据打平 (Flatten)
    # Shape: (Batch_Size * Group_Size, )
    rewards_tensor = torch.cat(all_final_advantages)
    # Keep interface consistent with other reward functions:
    # allow config-level reward scaling for GDPO as well.
    rewards_tensor = rewards_tensor * float(scale)
    valid_mask_tensor = torch.cat(valid_masks)

    
    # scores_dict 这里存的是 Sigmoid 后的 score，用于训练/评估里的 *_score 统计
    scores_dict = flatten_scores_dict(all_log_scores, task, use_similarity)
    return {
        "reward": rewards_tensor,   # 最终给 RL 的 Advantage (Z-scores)
        "scores": scores_dict,      # 原始物理性质 (用于 wandb)
        "valid_mask": valid_mask_tensor,
        "source_properties": src_props_list,
        "target_properties": tgt_prop_logs
    }

# 辅助函数 (保持不变)
def flatten_scores_dict(scores_dict, task, use_similarity, device='cpu'):
    keys = task.split('+')
    if use_similarity:
        keys.append('sim')
    parsed_data = {k: [] for k in keys}
    
    for group in scores_dict:       
        for sample in group:
            for k in keys:
                parsed_data[k].append(sample.get(k, 0.0))
                
    tensor_scores = {
        k: torch.tensor(v, dtype=torch.float32, device=device) 
        for k, v in parsed_data.items()
    }
    return tensor_scores
