from rdkit import Chem
from rdkit.Chem import QED, Crippen
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import Descriptors
import networkx as nx
import pandas as pd
from .sascores import calculateScore

import requests


def morgan_fp(mol, radius=2, nBits=2048):
    return GetMorganFingerprintAsBitVect(mol, radius, nBits)

def penalized_logp(mol):
    if mol is None: return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p  + normalized_SA + normalized_cycle

def compute_chem_properties_batch(source_smiles: list, target_smiles: list[list], task=None, admet_model=None):
    """
    Inputs:
        - source_smiles: List[str], SMILES strings for source molecules
        - target_smiles: List[List[str]], SMILES strings for generated molecules corresponding to each source
    Returns:
        - source_props: List[dict], property dictionary for each source molecule
        - target_props: List[List[dict]], property dictionaries for each list of target molecules
    """
    task_keys = set(task.lower().split("+")) if task else set()
    B = len(source_smiles)

    # Combine all SMILES strings: sources + all targets flattened
    flat_targets = [tgt for tgts in target_smiles for tgt in tgts]
    all_smiles = source_smiles + flat_targets

    # Generate RDKit Mol objects
    mols = [Chem.MolFromSmiles(smi) if smi is not None else None for smi in all_smiles]
    # RDKit may return an empty Mol (0 atoms) for malformed edge cases (e.g., empty SMILES).
    # Treat those as invalid to avoid downstream descriptor failures.
    valids = [mol is not None and mol.GetNumAtoms() > 0 for mol in mols]
    fps = [morgan_fp(mol, 2) if mol else None for mol in mols]

    # Predict ADMET properties in batch
    valid_smiles = [s for s, v in zip(all_smiles, valids) if v]
    admet_df = admet_model.predict(valid_smiles) if admet_model and ({"mutagenicity", "bbbp", "hia", "herg", "liv", "carc", "amp"} & task_keys) else pd.DataFrame()
    admet_idx = 0

    from .drd2 import get_score

    # print(f"ADMET DataFrame columns: {admet_df.columns}")

    source_props = []
    target_props = [[] for _ in range(B)]

    for i in range(len(all_smiles)):
        smi = all_smiles[i]
        mol = mols[i]
        fp = fps[i]
        is_valid = valids[i]
        r = {"valid": is_valid, "smiles": smi}

        if not is_valid:
            if i < B:
                source_props.append(r)
            else:
                target_props[(i - B) // len(target_smiles[0])].append(r)
            continue

        # Basic properties
        if "qed" in task_keys:
            r["qed"] = QED.qed(mol)
        if "plogp" in task_keys:
            r["plogp"] = penalized_logp(mol)
        if "drd2" in task_keys:
            r["drd2"] = get_score(smi)
        
        # ADMET properties
        if not admet_df.empty:
            row = admet_df.iloc[admet_idx]
            if "mutagenicity" in task_keys or "mutag" in task_keys:
                r["mutag"] = row.get("AMES", None)
            if "bbbp" in task_keys:
                r["bbbp"] = row.get("BBB_Martins", None)
            if "hia" in task_keys:
                r["hia"] = row.get("HIA_Hou", None)
            if "herg" in task_keys:
                r["herg"] = row.get("hERG", None)
            if "carc" in task_keys:
                r["carc"] = row.get("Carcinogens_Lagunin", None)
            if "amp" in task_keys:
                r["amp"] = row.get("PAMPA_NCATS", None)
            if "liv" in task_keys:
                r["liv"] = row.get("DILI", None)
            admet_idx += 1

        # Compute similarity if target
        if i >= B :
            group_idx = (i - B) // len(target_smiles[0])
            src_fp = fps[group_idx]  # source mol's fingerprint
            if src_fp and fp:
                r["sim"] = TanimotoSimilarity(fp, src_fp)
            else:
                r["sim"] = None
            target_props[group_idx].append(r)
        else:
            source_props.append(r)

    return source_props, target_props



def compute_test_chem_properties_batch(source_smiles: list, target_smiles: list[list], task=None, admet_model=None):
    """
    Inputs:
        - source_smiles: List[str], SMILES strings for source molecules
        - target_smiles: List[List[str]], SMILES strings for generated molecules corresponding to each source
    Returns:
        - source_props: List[dict], property dictionary for each source molecule
        - target_props: List[List[dict]], property dictionaries for each list of target molecules
    """
    task_keys = set(task.lower().split("+")) if task else set()
    B = len(source_smiles)

    # Combine all SMILES strings: sources + all targets flattened
    flat_targets = [tgt for tgts in target_smiles for tgt in tgts]
    all_smiles = source_smiles + flat_targets

    # Generate RDKit Mol objects
    mols = [Chem.MolFromSmiles(smi) if smi is not None else None for smi in all_smiles]
    # RDKit may return an empty Mol (0 atoms) for malformed edge cases (e.g., empty SMILES).
    # Treat those as invalid to avoid downstream descriptor failures.
    valids = [mol is not None and mol.GetNumAtoms() > 0 for mol in mols]
    fps = [morgan_fp(mol, 2) if mol else None for mol in mols]

    # Predict ADMET properties in batch
    valid_smiles = [s for s, v in zip(all_smiles, valids) if v]
    admet_df = admet_model.predict(valid_smiles) if admet_model and ({"mutagenicity", "bbbp", "hia", "herg", "liv", "carc", "amp"} & task_keys) else pd.DataFrame()
    admet_idx = 0

    from .drd2 import get_score

    # print(f"ADMET DataFrame columns: {admet_df.columns}")

    source_props = []
    target_props = [[] for _ in range(B)]

    for i in range(len(all_smiles)):
        smi = all_smiles[i]
        mol = mols[i]
        fp = fps[i]
        is_valid = valids[i]
        r = {"valid": is_valid, "smiles": smi}


        if not is_valid:
            if i < B:
                source_props.append(r)
            else:
                target_props[(i - B) // len(target_smiles[0])].append(r)
            continue

        # Basic properties
        r['sas'] = calculateScore(mol)
        if "qed" in task_keys:
            r["qed"] = QED.qed(mol)
        if "plogp" in task_keys:
            r["plogp"] = penalized_logp(mol)
        if "drd2" in task_keys:
            r["drd2"] = get_score(smi)
        
        # ADMET properties
        if not admet_df.empty:
            row = admet_df.iloc[admet_idx]
            if "mutagenicity" in task_keys or "mutag" in task_keys:
                r["mutag"] = row.get("AMES", None)
            if "bbbp" in task_keys:
                r["bbbp"] = row.get("BBB_Martins", None)
            if "hia" in task_keys:
                r["hia"] = row.get("HIA_Hou", None)
            if "herg" in task_keys:
                r["herg"] = row.get("hERG", None)
            if "carc" in task_keys:
                r["carc"] = row.get("Carcinogens_Lagunin", None)
            if "amp" in task_keys:
                r["amp"] = row.get("PAMPA_NCATS", None)
            if "liv" in task_keys:
                r["liv"] = row.get("DILI", None)
            admet_idx += 1

        # Compute similarity if target
        if i >= B :
            group_idx = (i - B) // len(target_smiles[0])
            src_fp = fps[group_idx]  # source mol's fingerprint
            if src_fp and fp:
                r["sim"] = TanimotoSimilarity(fp, src_fp)
            else:
                r["sim"] = None
            target_props[group_idx].append(r)
        else:
            source_props.append(r)

    return source_props, target_props
    



def compute_chem_properties_remote(source_smiles, target_smiles, task='plogp+qed+hia+drd2'):
    """Call ADMET service to compute properties"""
    
    url = 'http://localhost:5000/compute'
    
    data = {
        'source_smiles': source_smiles,
        'target_smiles': target_smiles,
        'task': task
    }
    
    try:
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            src_props = response.json().get('src_props', [])
            tgt_props = response.json().get('tgt_props', [])
            return src_props, tgt_props
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Request failed: {e}")
        return None
