from typing import Dict, List, TypedDict

import inspect
import torch


class EvalResult(TypedDict):
    rewards: List[List[float]]
    target_smiles: List[List[str]]
    target_smiles_properties: List[List[dict]]
    source_smiles: List[str]
    prompt: List[str]
    num_generated: int


@torch.no_grad()
def eval_train_env(
    actor,
    env,
    tokenizer,
    reward_fn,
    num_return_sequences: int = 1,
    num_envs: int = 1,
    num_beams: int = 1,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 0.7,
    max_new_tokens: int = 100,
    do_sample: bool = False,
    eval_indices: List[int] = None,
) -> Dict:
    if eval_indices is None:
        eval_indices = list(range(num_envs))

    prompt_list = []
    source_smiles_list = []
    target_thresholds_batch = []
    for idx in eval_indices:
        env.index = idx
        env.create_sample()
        prompt_list.append(env.prompt)
        source_smiles_list.append(env.source_smiles)
        target_thresholds_batch.append(dict(getattr(env, "current_target_thresholds", {}) or {}))

    _, _, _, target_smiles, _ = actor.generate(
        prompt=prompt_list,
        num_return_sequences=num_return_sequences,
        num_envs=len(eval_indices),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
    )

    try:
        sig = inspect.signature(reward_fn)
        if "target_thresholds_batch" in sig.parameters:
            reward_dict = reward_fn(
                source_smiles_list,
                target_smiles,
                target_thresholds_batch=target_thresholds_batch,
            )
        else:
            reward_dict = reward_fn(source_smiles_list, target_smiles)
    except (TypeError, ValueError):
        reward_dict = reward_fn(source_smiles_list, target_smiles)
    return {
        "rewards": reward_dict["reward"],
        "target_smiles": target_smiles,
        "scores": reward_dict['scores'],
        "penalties": reward_dict.get('penalties'),
        "target_properties": reward_dict["target_properties"],
        "source_properties": reward_dict["source_properties"],
        "different_properties": reward_dict.get("different_properties"),
        "source_smiles": source_smiles_list,
        "prompt": prompt_list,
        "num_generated": num_return_sequences,
    }
