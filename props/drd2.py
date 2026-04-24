
import os

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx, v in fp.GetNonzeroElements().items():
        nidx = idx % size
        nfp[0, nidx] += int(v)
    return nfp
# Global model object
clf_model = None

# Load model only once
def load_model(model_path=None):
    global clf_model
    if clf_model is None:
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "clf_py310_safe.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ DrD2 Model Not Found: {model_path}")
        
        clf_model = joblib.load(model_path)
        # Patch missing fields if necessary
        clf_model._n_support = np.array([1655, 504], dtype=np.int32)             # 用 n_support_ 来补
        clf_model.__dict__["_probA"] = np.array([-6.21194521])
        clf_model.__dict__["_probB"] = np.array([0.88511078])
        clf_model.classes_ = np.array([0, 1])

# Score function: return DRD2 probability score from SMILES
def get_score(smile):
    global clf_model
    if clf_model is None:
        load_model()

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return 0.0

    fp = fingerprints_from_mol(mol)

    # fp = fp.reshape(1, -1).astype(np.float64)
    # print("fp shape:", fp.shape)
    # print("fp dtype:", fp.dtype)
    # print("support_vectors dtype:", clf_model.support_vectors_.dtype)
    # print("dual_coef dtype:", clf_model.dual_coef_.dtype)
    
    # # 检查是否有其他属性
    # if hasattr(clf_model, '_sparse'):
    #     print("Is sparse:", clf_model._sparse)
    # if hasattr(clf_model, 'n_support_'):
    #     print("n_support dtype:", clf_model.n_support_.dtype)
    #try:
    return float(clf_model.predict_proba(fp)[:, 1])
    # except Exception:
    #     margin = clf_model.decision_function(fp)
    #     return float(1 / (1 + np.exp(-margin)))
        
