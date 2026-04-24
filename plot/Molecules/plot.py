cd /work/nvme/bfuy/rgao7/Molo
conda activate molo
PYTHONPATH=/work/nvme/bfuy/rgao7 python - <<'PY'
import os
import torch
import types, argparse, builtins, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

from Molo.utils import load_model_tokenizer, load_dataset_by_task
from Molo.rl.env import MoloEnv
from Molo.rl.ppo.actor import MoloActor
from Molo.props import compute_test_chem_properties_batch

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
    lora_adapter_path="runs/bpq/grpo/20260305_192549/best_adapters/lora_adapter",
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
    props_range={"sim":[0.6,1.0], "plogp":[2.5,5.0], "bbbp":[0.8,1.0], "qed":[0.8,1.0]},
    reward_weight={"qed":1.0, "plogp":1.0, "bbbp":1.0, "sim":1.2},
    include_delta=False,
    include_weight=False,
    random_sample=False,
    device=device,
)

prompt, source_smiles, _, _ = env.reset()
_, _, _, target_smiles, _ = actor.generate(
    prompt=[prompt],
    num_envs=1,
    num_return_sequences=4,
    temperature=0.7,
    max_new_tokens=100,
    top_k=50,
    top_p=0.95,
    do_sample=True,
)
targets = target_smiles[0][:4]

src_props_list, tgt_props_list = compute_test_chem_properties_batch(
    [source_smiles], [targets], task="bbbp+plogp+qed", admet_model=admet_model
)
src = src_props_list[0]
tgt = tgt_props_list[0]

mols = [Chem.MolFromSmiles(source_smiles)] + [Chem.MolFromSmiles(s) for s in targets]
legends = [
    f"source\n{source_smiles}\nqed={src.get('qed',0):.3f}, plogp={src.get('plogp',0):.3f}, bbbp={src.get('bbbp',0):.3f}"
]
for i, (smi, p) in enumerate(zip(targets, tgt), 1):
    legends.append(
        f"target_{i}\n{smi}\nqed={p.get('qed',0):.3f}, plogp={p.get('plogp',0):.3f}, "
        f"bbbp={p.get('bbbp',0):.3f}, sim={p.get('sim',0):.3f}"
    )

img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(700, 450), legends=legends, useSVG=False)
svg = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(700, 450), legends=legends, useSVG=True)

os.makedirs("plot", exist_ok=True)
img.save("plot/bpq_training_example.png")
with open("plot/bpq_training_example.svg", "w", encoding="utf-8") as f:
    f.write(svg)

print("SOURCE:", source_smiles)
print("TARGETS:", targets)
print("Saved: plot/bpq_training_example.png")
print("Saved: plot/bpq_training_example.svg")
PY
