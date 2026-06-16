"""
solubility_api.py
=================
FastAPI wrapper for the real XGBoost solubility model from:
https://github.com/ali1810/Water_Solubility

This is the ACTUAL model — XGBoost trained on curated dataset.
Features: 125 RDKit descriptors + 128 Morgan FP + 7 functional groups + 38 structural = 298 total

Deploy on Hugging Face Spaces:
  1. Copy this file to your Space
  2. Copy xgboost_model_298_4045.json to your Space
  3. pip install fastapi uvicorn rdkit xgboost numpy pandas
  4. uvicorn solubility_api:app --host 0.0.0.0 --port 7860

Then call from your HTML app:
  POST https://YOUR-SPACE.hf.space/predict
  {"smiles": "CCO"}

Author: Dr. Mushtaq Ali · KIT
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd
import json
import os

# ── RDKIT ────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen
    from rdkit.Chem import AllChem
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("WARNING: RDKit not available")

# ── XGBOOST ──────────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("WARNING: XGBoost not available")

# ── LOAD MODEL & FEATURE ORDER ───────────────────────────────────────────────
MODEL = None
FEATURE_ORDER = None
MODEL_PATH = os.environ.get("MODEL_PATH", "xgboost_model_298_4045.json")
FEATURE_ORDER_PATH = os.environ.get("FEATURE_ORDER_PATH", "feature_order.json")

app = FastAPI(
    title="Solubility Prediction API",
    description="""
    Real XGBoost solubility prediction model trained on curated dataset.
    Model: xgboost_model_298_4045.json
    Features: 125 RDKit + 128 Morgan FP + 7 FG + 38 structural = 298 total
    Source: github.com/ali1810/Water_Solubility
    Author: Dr. Mushtaq Ali, KIT
    """,
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def load_model():
    global MODEL, FEATURE_ORDER
    if XGB_OK and os.path.exists(MODEL_PATH):
        try:
            MODEL = xgb.XGBRegressor()
            MODEL.load_model(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Model load failed: {e}")

    if os.path.exists(FEATURE_ORDER_PATH):
        with open(FEATURE_ORDER_PATH) as f:
            FEATURE_ORDER = json.load(f)
        print(f"✅ Feature order loaded: {len(FEATURE_ORDER)} features")


# ════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — exact copy from Water-Solubility.py
# ════════════════════════════════════════════════════════════════════════════

def get_charges(smiles):
    return 1 if '+' in smiles else (-1 if '-' in smiles else 0)

def get_many_double_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None: return 0
    count = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.DOUBLE)
    return 1 if count > 4 else 0

def get_atom_degrees(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None: return np.zeros(7, dtype=int)
    mol = Chem.AddHs(mol)
    vec = np.zeros(7)
    for bond in mol.GetBonds():
        for atom in [bond.GetBeginAtom(), bond.GetEndAtom()]:
            d = atom.GetDegree()
            if d < 7: vec[d] += 1
    return vec.astype(int)

def get_atom_valences(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros(7, dtype=int)
    mol = Chem.AddHs(mol)
    vec = np.zeros(7)
    for bond in mol.GetBonds():
        for atom in [bond.GetBeginAtom(), bond.GetEndAtom()]:
            v = atom.GetTotalValence()
            if v < 7: vec[v] += 1
    return vec.astype(int)

def get_atom_hybridization(smiles):
    hybs = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros(7, dtype=int)
    mol = Chem.AddHs(mol)
    vec = np.zeros(7)
    for bond in mol.GetBonds():
        for atom in [bond.GetBeginAtom(), bond.GetEndAtom()]:
            for i, h in enumerate(hybs):
                if atom.GetHybridization() == h: vec[i] += 1
    return vec.astype(int)

def get_aromatic_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return 0
    mol = Chem.AddHs(mol)
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetIsAromatic(): count += 1
        if bond.GetEndAtom().GetIsAromatic(): count += 1
    return count

def get_bond_types(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros(5, dtype=int)
    mol = Chem.AddHs(mol)
    vec = np.zeros(5)
    types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'ZERO']
    for bond in mol.GetBonds():
        bt = bond.GetBondType().name
        if bt in types: vec[types.index(bt)] += 1
    return vec.astype(int)

def is_conjugated(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return 0
    mol = Chem.AddHs(mol)
    return sum(1 for b in mol.GetBonds() if b.GetIsConjugated())

def get_bonds_in_ring(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return 0
    mol = Chem.AddHs(mol)
    return len(Chem.GetSymmSSSR(mol))

def get_bond_chirality(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros(4, dtype=int)
    mol = Chem.AddHs(mol)
    stereos = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
               Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]
    vec = np.zeros(4)
    for bond in mol.GetBonds():
        s = bond.GetStereo()
        for i, st in enumerate(stereos):
            if s == st: vec[i] += 1
    return vec.astype(int)

def generate_features38(smiles):
    """38 structural features — from Water-Solubility.py"""
    columns = [
        'charge','many_double_bonds','atoms_degree_0','atoms_degree_1','atoms_degree_2',
        'atoms_degree_3','atoms_degree_4','atoms_degree_5','atoms_degree_6',
        'atoms_valence_0','atoms_valence_1','atoms_valence_2','atoms_valence_3',
        'atoms_valence_4','atoms_valence_5','atoms_valence_6',
        'atom_hybridization_S','atom_hybridization_SP','atom_hybridization_SP2',
        'atom_hybridization_SP3','atom_hybridization_SP3D','atom_hybridization_SP3D2',
        'atom_hybridization_UNSPECIFIED','aromatic_atoms','single_bonds','double_bonds',
        'triple_bonds','aromatic_bonds','zero_bonds','conjugated_bonds','bonds_in_ring',
        'chirality_none','chirality_any','chirality_z','chirality_e',
        'n_atoms','n_bonds','n_rings'
    ]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.DataFrame([np.zeros(38)], columns=columns)

    mol_h = Chem.AddHs(mol)
    row = [
        get_charges(smiles),
        get_many_double_bonds(smiles),
        *get_atom_degrees(smiles).tolist(),
        *get_atom_valences(smiles).tolist(),
        *get_atom_hybridization(smiles).tolist(),
        get_aromatic_atoms(smiles),
        *get_bond_types(smiles).tolist(),
        is_conjugated(smiles),
        get_bonds_in_ring(smiles),
        *get_bond_chirality(smiles).tolist(),
        mol_h.GetNumAtoms(),
        mol_h.GetNumBonds(),
        len(Chem.GetSymmSSSR(mol_h))
    ]
    return pd.DataFrame([row], columns=columns)

def calculate_rdkit_features(smiles):
    """125 RDKit molecular descriptors"""
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return pd.DataFrame()
    descriptor_names = [d[0] for d in Descriptors._descList]
    descriptor_funcs = [d[1] for d in Descriptors._descList]
    features = []
    for fn in descriptor_funcs:
        try: features.append(fn(mol))
        except: features.append(0.0)
    df = pd.DataFrame([features], columns=descriptor_names)
    return df.iloc[:, :125]

def fingerprint_128(smiles):
    """128-bit Morgan fingerprint (radius=2)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.DataFrame([np.zeros(128, dtype=int)])
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
    arr = np.zeros(128, dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return pd.DataFrame([arr])

def get_functional_groups(smiles):
    """7 functional group features"""
    fg = {
        'Hydroxyl Group': '[OH]',
        'Carbonyl Group': 'C=O',
        'Amide Group': 'C(=O)N',
        'Carboxyl Group': 'C(=O)[OH]',
        'Alkyl': '[R]',
        'Aromatic Rings': 'c',
        'Alkene': 'C=C',
    }
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.DataFrame([{k: 0 for k in fg}])
    result = {}
    for name, smarts in fg.items():
        try:
            pat = Chem.MolFromSmarts(smarts)
            result[name] = 1 if (pat and mol.HasSubstructMatch(pat)) else 0
        except:
            result[name] = 0
    return pd.DataFrame([result])

def calc_mol_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol else None

def clean_features(df):
    """Replace NaN/Inf with safe values — from Water-Solubility.py"""
    arr = np.array(df, dtype=np.float64)
    nan_mask = np.isnan(arr)
    if np.any(nan_mask):
        mean = np.nanmean(arr) if not np.all(nan_mask) else 0.0
        arr[nan_mask] = mean
    inf_mask = np.isinf(arr)
    arr[inf_mask] = np.sign(arr[inf_mask]) * 1e6
    arr = np.clip(arr, -1e6, 1e6)
    return pd.DataFrame(arr, columns=df.columns)

def compute_all_features(smiles: str) -> pd.DataFrame:
    """
    Compute all 298 features exactly as in Water-Solubility.py:
    125 RDKit + 128 Morgan FP + 7 FG + 38 structural = 298
    """
    df125 = calculate_rdkit_features(smiles)
    df128 = fingerprint_128(smiles)
    df7   = get_functional_groups(smiles)
    df38  = generate_features38(smiles)
    combined = pd.concat([df125, df128, df7, df38], axis=1)
    return clean_features(combined)

def applicability_domain(smiles: str) -> dict:
    """
    Applicability domain check.
    Currently uses MW < 600 as per the active code in Water-Solubility.py.
    The full t-SNE/Mahalanobis code is commented out there — 
    this replicates the active implementation faithfully.
    """
    mw = calc_mol_weight(smiles)
    if mw is None:
        return {"in_domain": False, "domain_score": 0, "mw": None,
                "note": "Invalid SMILES — cannot compute MW"}
    in_domain = mw < 600
    # Score: 100 for MW=0, decreasing toward 0 at MW=800
    score = max(0, min(100, int(100 * (1 - max(0, mw - 100) / 700))))
    return {
        "in_domain": in_domain,
        "domain_score": score,
        "mw": round(mw, 2),
        "threshold": 600,
        "note": f"MW={mw:.1f} — {'within' if in_domain else 'outside'} applicability domain (MW < 600 threshold)"
    }

def classify_logS(logS: float) -> dict:
    """Classify logS into solubility class (Delaney classification)"""
    if logS > 0:
        return {"class": "highly_soluble", "label": "Highly Soluble",
                "color": "blue", "range": "logS > 0"}
    elif logS > -2:
        return {"class": "soluble", "label": "Soluble",
                "color": "green", "range": "-2 < logS ≤ 0"}
    elif logS > -4:
        return {"class": "slightly_soluble", "label": "Slightly Soluble",
                "color": "amber", "range": "-4 < logS ≤ -2"}
    else:
        return {"class": "practically_insoluble", "label": "Practically Insoluble",
                "color": "red", "range": "logS ≤ -4"}


# ════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    smiles: str
    include_features: bool = False  # Return full feature vector (298 values)

class PredictResponse(BaseModel):
    smiles: str
    valid: bool
    logS: Optional[float]
    logS_rounded: Optional[float]
    mol_per_liter: Optional[float]
    gram_per_liter: Optional[float]
    molecular_weight: Optional[float]
    solubility_class: Optional[str]
    solubility_label: Optional[str]
    in_domain: bool
    domain_score: int
    domain_note: str
    model: str
    n_features: int
    features_used: dict
    error: Optional[str] = None

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Predict water solubility (logS) from SMILES.

    Uses the REAL XGBoost model (xgboost_model_298_4045.json) with 298 features:
    - 125 RDKit molecular descriptors
    - 128-bit Morgan fingerprints (radius=2)
    - 7 functional group features
    - 38 structural features (atom degrees, valences, hybridisation, bond types)
    """
    smiles = req.smiles.strip()

    # Validate SMILES
    if not RDKIT_OK:
        raise HTTPException(500, "RDKit not available on this server")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return PredictResponse(
            smiles=smiles, valid=False, logS=None, logS_rounded=None,
            mol_per_liter=None, gram_per_liter=None, molecular_weight=None,
            solubility_class=None, solubility_label=None,
            in_domain=False, domain_score=0,
            domain_note="Invalid SMILES string",
            model="xgboost_model_298_4045", n_features=298,
            features_used={"rdkit": 125, "morgan_fp": 128, "functional_groups": 7, "structural": 38},
            error="Invalid SMILES — RDKit could not parse"
        )

    # Compute features
    try:
        combined_df = compute_all_features(smiles)
    except Exception as e:
        raise HTTPException(500, f"Feature computation failed: {str(e)}")

    # Predict
    logS = None
    if MODEL is not None:
        try:
            logS = float(MODEL.predict(combined_df)[0])
        except Exception as e:
            raise HTTPException(500, f"Model prediction failed: {str(e)}")
    else:
        raise HTTPException(503, "Model not loaded — check MODEL_PATH environment variable")

    # Post-process
    mw = calc_mol_weight(smiles)
    mol_per_liter = float(10 ** logS)
    gram_per_liter = mol_per_liter * (mw or 0)
    domain = applicability_domain(smiles)
    sol_class = classify_logS(logS)

    return PredictResponse(
        smiles=smiles,
        valid=True,
        logS=logS,
        logS_rounded=round(logS, 2),
        mol_per_liter=round(mol_per_liter, 4),
        gram_per_liter=round(gram_per_liter, 4),
        molecular_weight=mw,
        solubility_class=sol_class["class"],
        solubility_label=sol_class["label"],
        in_domain=domain["in_domain"],
        domain_score=domain["domain_score"],
        domain_note=domain["note"],
        model="xgboost_model_298_4045",
        n_features=combined_df.shape[1],
        features_used={
            "rdkit_descriptors": 125,
            "morgan_fingerprint_128bit": 128,
            "functional_groups": 7,
            "structural_features": 38,
            "total": combined_df.shape[1]
        }
    )


@app.post("/predict_batch")
async def predict_batch(smiles_list: List[str]):
    """Predict solubility for a list of SMILES (max 100)."""
    if len(smiles_list) > 100:
        raise HTTPException(400, "Maximum 100 SMILES per batch request")
    results = []
    for smi in smiles_list:
        try:
            r = await predict(PredictRequest(smiles=smi))
            results.append(r.dict())
        except Exception as e:
            results.append({"smiles": smi, "error": str(e)})
    return results


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "rdkit_available": RDKIT_OK,
        "xgboost_available": XGB_OK,
        "model_file": MODEL_PATH,
        "feature_order_loaded": FEATURE_ORDER is not None,
        "description": "Real XGBoost solubility model from ali1810/Water_Solubility",
        "features": {
            "rdkit_descriptors": 125,
            "morgan_fingerprints_128bit": 128,
            "functional_groups": 7,
            "structural_features": 38,
            "total": 298
        },
        "applicability_domain": "MW < 600 threshold (active) + t-SNE/Mahalanobis (development)",
        "live_streamlit": "https://aqua-solubility-prediction.streamlit.app/"
    }


@app.get("/model_info")
async def model_info():
    return {
        "model_type": "XGBoost Regressor",
        "model_file": "xgboost_model_298_4045.json",
        "target": "logS (log10 of molar aqueous solubility)",
        "n_features": 298,
        "feature_groups": {
            "RDKit molecular descriptors (125)": [
                "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
                "NumRotatableBonds", "RingCount", "... 118 more"
            ],
            "Morgan fingerprints 128-bit radius=2 (128)": [
                "Circular fingerprint encoding local atom environments"
            ],
            "Functional groups (7)": [
                "Hydroxyl Group", "Carbonyl Group", "Amide Group",
                "Carboxyl Group", "Alkyl", "Aromatic Rings", "Alkene"
            ],
            "Structural features (38)": [
                "charge", "many_double_bonds",
                "atoms_degree_0..6", "atoms_valence_0..6",
                "atom_hybridization (7 types)", "aromatic_atoms",
                "bond_types (single/double/triple/aromatic/zero)",
                "conjugated_bonds", "bonds_in_ring",
                "bond_chirality (4 types)", "n_atoms", "n_bonds", "n_rings"
            ]
        },
        "training_data": "Curated dataset (final_unique_train.csv)",
        "test_data": "final_unique_test.csv",
        "evaluation": "See model_evaluation_test_data.ipynb",
        "applicability_domain": {
            "active_method": "Molecular weight threshold (MW < 600)",
            "development_method": "t-SNE + PCA + Mahalanobis distance (commented out in source)"
        },
        "output": {
            "logS": "log10(mol/L) — negative = less soluble",
            "mol_per_liter": "10^logS",
            "gram_per_liter": "mol/L × molecular weight",
            "solubility_classes": {
                "highly_soluble": "logS > 0",
                "soluble": "-2 < logS ≤ 0",
                "slightly_soluble": "-4 < logS ≤ -2",
                "practically_insoluble": "logS ≤ -4"
            }
        },
        "source": "https://github.com/ali1810/Water_Solubility",
        "author": "Dr. Mushtaq Ali, KIT",
        "publication": "doi:10.1021/acs.jcim.4c02399"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
