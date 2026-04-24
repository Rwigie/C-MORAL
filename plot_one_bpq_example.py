import os
import torch
import types, argparse, builtins, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

from Molo.utils import load_model_tokenizer, load_dataset_by_task
from Molo.rl.env import MoloEnv
from Molo.rl.ppo.actor import MoloActor
from Molo.rl.reward_sigmoid import compute_reward_sigmoid
from Molo.props import MOLO_PROPERTIES, compute_test_chem_properties_batch
from Molo.test.test_metric import PROP_THETA

RDLogger.DisableLog("rdApp.*")

safe_types = [
    argparse.Namespace, np.ndarray, np.dtype, np.core.multiarray._reconstruct,
    np.dtype("float64").__class__, set, slice, tuple, list, dict, float, int,
    str, bytes, type, types.SimpleNamespace, builtins.object,
]
with torch.serialization.safe_globals(safe_types):
    from admet_ai import ADMETModel
    admet_model = ADMETModel(num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"

model, tokenizer = load_model_tokenizer(
    base_model="Mistral",
    base_model_path="models/Mistral/base",
    lora_adapter_path="models/Mistral/gellmoC-lora",
    use_lora=True,
    device=device,
)
actor = MoloActor(model, tokenizer, device=device)

dataset = load_dataset_by_task(
    task="bbbp+plogp+qed",
    save_path="data_new/mumo_by_task",
    split="train",
    full_save_path="data_new/MumoInstruct_arrow",
)

env = MoloEnv(
    dataset=dataset,
    task="bbbp+plogp+qed",
    props_delta={"qed": 0.1, "plogp": 1.0, "bbbp": 0.1},
    props_range={},
    reward_weight={"qed":1.0, "plogp":1.0, "bbbp":1.0, "sim":1.2},
    include_delta=False,
    include_weight=False,
    random_sample=False,
    device=device,
)

task_keys = ["bbbp", "plogp", "qed"]
selected_index = None
source_smiles = None
prompt = None
src = None
selected_near = []
selected_sub = []

for idx in range(len(dataset)):
    env.index = idx
    env.create_sample()
    candidate_smiles = env.source_smiles
    src_props_list, _ = compute_test_chem_properties_batch(
        [candidate_smiles], [[candidate_smiles]], task="bbbp+plogp+qed", admet_model=admet_model
    )
    candidate_src = src_props_list[0]

    near = []
    sub = []
    for k in task_keys:
        val = candidate_src.get(k)
        if val is None:
            continue
        info = MOLO_PROPERTIES[k]
        theta = PROP_THETA.get(k, info.target_threshold)
        if info.optimization_direction == "maximize":
            if val >= theta:
                near.append(k)
            else:
                sub.append(k)
        else:
            if val <= theta:
                near.append(k)
            else:
                sub.append(k)

    # pick a source where:
    # 1) plogp is NOT near-optimal
    # 2) at least one of qed/bbbp is near-optimal
    if ("plogp" in sub) and (("qed" in near) or ("bbbp" in near)):
        selected_index = idx
        source_smiles = candidate_smiles
        prompt = env.prompt
        src = candidate_src
        selected_near = near
        selected_sub = sub
        break

if selected_index is None:
    raise RuntimeError("No source molecule found with qed or bbbp near-optimal while plogp is sub-optimal.")

_, _, _, target_smiles, _ = actor.generate(
    prompt=[prompt],
    num_envs=1,
    num_return_sequences=4,
    temperature=1,
    max_new_tokens=100,
    top_k=100,
    top_p=0.98,
    do_sample=True,
)
targets = [s for s in target_smiles[0] if isinstance(s, str) and s.strip()][:4]
if not targets:
    raise RuntimeError("No valid target SMILES generated.")

src_props_list, tgt_props_list = compute_test_chem_properties_batch(
    [source_smiles], [targets], task="bbbp+plogp+qed", admet_model=admet_model
)
src = src_props_list[0]
tgt = tgt_props_list[0]

reward_dict = compute_reward_sigmoid(
    source_smiles=[source_smiles],
    target_smiles=[targets],
    task="bbbp+plogp+qed",
    reward_mode="absolute",
    aggregation="weighted_sum",
    reward_weight={"qed": 1.0, "plogp": 1.0, "bbbp": 1.0, "sim": 1.2},
    use_similarity=True,
    use_props_delta=True,
    props_delta={"qed": 0.1, "plogp": 1.0, "bbbp": 0.1},
    props_delta_penalty={"qed": -0.7, "plogp": -0.7, "bbbp": -0.7, "sim": -1.0},
    use_props_range=False,
    props_range={},
    props_range_penalty={"sim": -1.0, "plogp": -1.0, "bbbp": -1.0, "qed": -1.0},
    scale=2.0,
    clip=5.0,
    admet_model=admet_model,
)
reward_tensor = reward_dict["reward"]
scores_dict = reward_dict["scores"]
overall_reward_mean = float(reward_tensor.mean().item())
overall_reward_max = float(reward_tensor.max().item())

out_dir = "plot/bpq_training_example_split"
os.makedirs(out_dir, exist_ok=True)

src_mol = Chem.MolFromSmiles(source_smiles)
if src_mol is not None:
    Draw.MolToFile(src_mol, os.path.join(out_dir, "source.png"), size=(1200, 900))

for i, smi in enumerate(targets[:4], 1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        Draw.MolToFile(mol, os.path.join(out_dir, f"target_{i}.png"), size=(1200, 900))

txt_path = os.path.join(out_dir, "labels.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"selected_dataset_index: {selected_index}\n")
    f.write(f"selected_near_optimal: {selected_near}\n")
    f.write(f"selected_sub_optimal: {selected_sub}\n")
    f.write(f"source_smiles: {source_smiles}\n")
    f.write(
        f"source_props: qed={src.get('qed',0):.4f}, "
        f"plogp={src.get('plogp',0):.4f}, "
        f"bbbp={src.get('bbbp',0):.4f}\n\n"
    )
    f.write(f"overall_reward_mean: {overall_reward_mean:.6f}\n")
    f.write(f"overall_reward_max : {overall_reward_max:.6f}\n\n")
    for i, (smi, p) in enumerate(zip(targets[:4], tgt[:4]), 1):
        idx = i - 1
        reward_i = float(reward_tensor[idx].item())
        score_i = {k: float(v[idx].item()) for k, v in scores_dict.items()}
        f.write(f"target_{i}_smiles: {smi}\n")
        f.write(
            f"target_{i}_props: qed={p.get('qed',0):.4f}, "
            f"plogp={p.get('plogp',0):.4f}, "
            f"bbbp={p.get('bbbp',0):.4f}, "
            f"sim={p.get('sim',0):.4f}\n"
        )
        f.write(
            "target_{}_scores: ".format(i)
            + ", ".join([f"{k}={v:.6f}" for k, v in score_i.items()])
            + "\n"
        )
        f.write(f"target_{i}_reward: {reward_i:.6f}\n\n")

print("SOURCE:", source_smiles)
print("TARGETS:", targets)
print("Saved dir:", out_dir)
print("Saved text:", txt_path)
