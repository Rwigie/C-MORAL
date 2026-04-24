import numpy as np
import torch
from rdkit import Chem
from ..props import compute_chem_properties_remote, MOLO_PROPERTIES, compute_chem_properties_batch

def compute_sigmoid_score(val, target, k=10.0, direction="maximize"):
    """
    单边 Sigmoid (S型曲线)：用于 'Improvement' 任务。
    逻辑：必须超过(或低于) target 才能获得高分。
    """
    if val is None: return 1e-6
    
    # 截断防止溢出
    val = np.clip(val, -100, 100)
    
    if direction == "maximize":
        # 目标: Val > Target (越大越好)
        # formula: 1 / (1 + exp(-k * (x - T)))
        return 1 / (1 + np.exp(-k * (val - target)))
    elif direction == "minimize":
        # 目标: Val < Target (越小越好)
        # formula: 1 / (1 + exp(k * (x - T)))
        return 1 / (1 + np.exp(k * (val - target)))
    else:
        return 0.0

def compute_double_sigmoid_score(val, target_low, target_high, k=10.0):
    """
    双边 Sigmoid (钟形曲线)：用于 'Maintenance' (维稳) 任务。
    逻辑：必须落在 [target_low, target_high] 区间内，太高太低都扣分。
    本质是两个 Sigmoid 的乘积。
    """
    if val is None: return 0.0

    # 1. 下限约束: 要求 val > target_low
    score_low = compute_sigmoid_score(val, target_low, k=k, direction="maximize")
    
    # 2. 上限约束: 要求 val < target_high
    score_high = compute_sigmoid_score(val, target_high, k=k, direction="minimize")
    
    # 3. 几何融合
    return score_low * score_high

def geometric_mean(scores, task_keys):
    product = 1.0
    valid_keys_count = 0
    
    for k in task_keys:
        s = scores.get(k, 0.0)
        # 加一个小 epsilon 防止 0 分导致梯度彻底消失
        product *= (s + 1e-6)
        valid_keys_count += 1
        
    if valid_keys_count > 0:
        final_reward = product ** (1.0 / valid_keys_count)
    else:
        final_reward = 0.0
    
    return final_reward

def weighted_sum_mean(scores, task_keys, reward_weight):
    final_reward = 0.0
    #total_weight = 0.0
    for k in task_keys:
        s = scores.get(k, 0.0)
        w = reward_weight.get(k, 1.0) if reward_weight else 1.0
        final_reward += s 
        #total_weight += w
    
    return final_reward 
        
def _resolve_target_threshold(prop_key, direction, source_props=None, target_thresholds=None, props_range=None):
    # 1) per-sample target threshold (if caller provides it)
    if isinstance(target_thresholds, dict) and prop_key in target_thresholds:
        return float(target_thresholds[prop_key]), "sample_target"

    # 2) fallback to static threshold in property metadata
    if prop_key in MOLO_PROPERTIES:
        return float(MOLO_PROPERTIES[prop_key].target_threshold), "static_theta"

    # 3) task-level range target from config (lowest priority)
    if isinstance(props_range, dict) and prop_key in props_range:
        rng = props_range.get(prop_key)
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
            return (lo if direction == "maximize" else hi), "range_target"
    return None, "none"


def calculate_ge_llm_reward(source_props, target_props, task_keys, reward_weight, target_thresholds=None, props_range=None):
    """
    计算基于 GeLLM4o-C 逻辑的综合 Reward。
    """
    scores = {}
    used_targets = {}
    
    for key in task_keys:
        prop_key = key.lower()
        val_new = target_props.get(prop_key)
        
        # --- 特殊处理 Similarity ---
        # Sim 通常总是 Maximize 任务，且没有 "Source Sim" (Source 对自己的 Sim 是 1.0)
        # 我们可以设定 Sim 的目标是 > 0.4 或 0.6
        if prop_key == 'sim':
            sim_target = 0.6 # 或者从外部配置传入
            # Sim 也是单边提升任务
            scores[key] = compute_sigmoid_score(val_new, sim_target, k=60.0, direction="maximize")
            continue

        # --- 获取属性元数据 ---
        if prop_key not in MOLO_PROPERTIES:
            scores[key] = 0.0 # 未知属性给0分
            continue
            
        prop_info = MOLO_PROPERTIES[prop_key]
        val_src = source_props.get(prop_key)
        
        if val_src is None or val_new is None:
            scores[key] = 0.0
            continue

        # 读取论文定义的参数 [cite: 95, 147]
        Theta = prop_info.target_threshold  # 药物相关水平 (及格线)
        Delta = prop_info.delta_threshold   # 要求的变化量/波动范围
        Direction = prop_info.optimization_direction
        dynamic_target, target_mode = _resolve_target_threshold(
            prop_key=prop_key,
            direction=Direction,
            source_props=source_props,
            target_thresholds=target_thresholds,
            props_range=props_range,
        )
        used_targets[prop_key] = {
            "target": dynamic_target,
            "mode": target_mode,
            "direction": Direction,
            "source": val_src,
            "generated": val_new,
        }
        
        # 动态调整 k (斜率)
        # 如果属性范围很大(如LogP -5~6)，k要小一点(如2.0)，否则梯度会消失
        # 如果属性范围小(如QED 0~1)，k要大一点(如10.0)，对微小变化敏感
        val_range = prop_info.value_range[1] - prop_info.value_range[0]
        # current_k = 10.0 if val_range <= 2.0 else 2.0
        current_k = 4.0 / Delta

        # --- 核心逻辑: 判断任务类型  ---
        
        # 判断 Source 是否已经 "Near-Optimal" (及格)
        is_near_optimal = False
        if Direction == "maximize":
            if val_src >= Theta: is_near_optimal = True
        elif Direction == "minimize":
            if val_src <= Theta: is_near_optimal = True
            
        # If a dynamic target exists, optimize directly toward that target first.
        # This keeps reward aligned with prompt targets when they are not static.
        if dynamic_target is not None and target_mode in ("sample_target", "range_target"):
            score = compute_sigmoid_score(val_new, dynamic_target, k=current_k, direction=Direction)
        elif is_near_optimal:
            # === Case A: 维稳任务 (Maintenance) ===
            # 目标: |New - Src| <= Delta 
            # 使用双边 Sigmoid
            low_bound = val_src - Delta
            high_bound = val_src + Delta
            if Direction == "maximize":
                # score = compute_double_sigmoid_score(val_new, target_low=val_src, target_high= high_bound, k=current_k)
                target_val = val_src
                score = compute_sigmoid_score(val_new, target_val, k=current_k, direction="maximize")
            else:
                target_val = val_src
                score = compute_sigmoid_score(val_new, target_val, k=current_k, direction="minimize")
            
        else:
            # === Case B: 提升任务 (Improvement) ===
            # 目标: 至少提升 Delta 
            # 使用单边 Sigmoid
            if Direction == "maximize":
                target_val = val_src + Delta
                score = compute_sigmoid_score(val_new, target_val, k=current_k, direction="maximize")
            else: 
                target_val = val_src - Delta
                score = compute_sigmoid_score(val_new, target_val, k=current_k, direction="minimize")
        
        scores[key] = score

    # --- 聚合 (Geometric Mean) ---
    # 几何平均能确保所有目标同时满足 (AND 逻辑)
    # Reward = (S1 * S2 * ... * Sn) ^ (1/n)
    
    final_reward = geometric_mean(scores, task_keys)
    #final_reward = weighted_sum_mean(scores, task_keys, reward_weight)

    return final_reward, scores, used_targets

def compute_reward_sigmoid(
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
    target_thresholds_batch=None,
):
    
    # 1. 解析任务 Key
    task_keys = set(task.lower().split("+"))
    if use_similarity:
        task_keys.add("sim")

    # 2. 预处理输入
    if isinstance(target_smiles[0], str):
        target_smiles = [[sm] for sm in target_smiles]

    # 3. 计算属性 (这是唯一耗时的步骤)
    # 返回: src_props (List[Dict]), tgt_props (List[List[Dict]])
    # src_props_list, tgt_props_list = compute_chem_properties_remote(source_smiles, target_smiles, task=task)
    src_props_list, tgt_props_list = compute_chem_properties_batch(source_smiles, target_smiles, task=task, admet_model=admet_model)

    # 4. 结果容器
    all_rewards = []
    all_scores = []
    valid_masks = []
    
    # 用于日志记录 (可选)
    src_prop_logs = src_props_list
    tgt_prop_logs = [] 

    # 5. 循环计算
    for sample_idx, (src_p, tgt_group) in enumerate(zip(src_props_list, tgt_props_list)):
        group_rewards = []
        group_scores = []
        tgt_prop_sublist = []
        sample_target_thresholds = None
        if isinstance(target_thresholds_batch, list) and sample_idx < len(target_thresholds_batch):
            sample_target_thresholds = target_thresholds_batch[sample_idx]

        for tgt_p in tgt_group:
            tgt_prop_sublist.append(tgt_p)
            
            # --- 处理无效分子 ---
            if not tgt_p.get("valid", False):
                # 无效分子直接给 0 分 (或者极低分)
                reward = -0
                group_rewards.append(reward)
                valid_masks.append(0.0)
                group_scores.append({k: 0.0 for k in task_keys})
                continue
            
            valid_masks.append(1.0)
            
            # --- 调用新的 Reward 逻辑 ---
            # 这里的 reward 已经在 [0, 1] 之间了
            reward, scores, used_targets = calculate_ge_llm_reward(
                source_props=src_p, 
                target_props=tgt_p, 
                task_keys=task_keys, 
                reward_weight=reward_weight,
                target_thresholds=sample_target_thresholds,
                props_range=props_range,
            )
            # if sample_idx < 2 and len(group_rewards) == 0:
            #     print(f"[DEBUG][reward] sample_idx={sample_idx} target_thresholds_batch={sample_target_thresholds}")
            #     print(f"[DEBUG][reward] sample_idx={sample_idx} used_targets={used_targets}")
            
            # 应用 Scale
            reward = reward * scale
            
            group_rewards.append(reward)
            # 保存 score 供 logging (这里的 score 也是归一化后的 [0,1])
            group_scores.append(scores)
        
        all_rewards.append(group_rewards)
        all_scores.append(group_scores)
        tgt_prop_logs.append(tgt_prop_sublist)

    # 6. 格式化输出 (Tensor化)
    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32).flatten() # Shape: (B,)
    valid_masks = torch.tensor(valid_masks, dtype=torch.float32).flatten()
    
    # 辅助函数 flatten_scores_dict 保持你原来的即可
    scores_dict = flatten_scores_dict(all_scores, task, use_similarity)

    return {
        "reward": rewards_tensor,
        "scores": scores_dict,
        "valid_mask": valid_masks,
        # 保留属性值供 Debug
        "source_properties": src_prop_logs, 
        "target_properties": tgt_prop_logs
    }

# 辅助函数保持你原来的，这里不需要变动
def flatten_scores_dict(scores_dict, task, use_similarity, device='cpu'):
    keys = task.split('+')
    if use_similarity: keys.append('sim')
    parsed_data = {k: [] for k in keys}
    
    for group in scores_dict:
        for sample in group:
            for k in keys:
                parsed_data[k].append(sample.get(k, 0.0))
                
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in parsed_data.items()}
