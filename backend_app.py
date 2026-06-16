"""
app.py — ChemPredict Unified FastAPI Backend
============================================
Deploy on Hugging Face Spaces (free tier).
Serves both XGBoost and MPNN solubility models.

Start locally:
  uvicorn app:app --reload --port 8000

Deploy on HuggingFace:
  Create a Space → upload this file + all model files
  → Space auto-builds and runs on port 7860
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

app = FastAPI(
    title="ChemPredict API",
    description="XGBoost + MPNN solubility prediction · Dr. Mushtaq Ali · KIT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins (GitHub Pages etc.)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
XGB_MODEL  = None
MPNN_MODEL = None

@app.on_event("startup")
async def load_models():
    global XGB_MODEL, MPNN_MODEL
    # XGBoost
    xgb_path = os.environ.get("XGB_PATH", "xgboost_model_298_4045.json")
    if os.path.exists(xgb_path):
        try:
            import xgboost as xgb
            XGB_MODEL = xgb.XGBRegressor()
            XGB_MODEL.load_model(xgb_path)
            print(f"✅ XGBoost loaded: {xgb_path}")
        except Exception as e:
            print(f"⚠ XGBoost failed: {e}")
    else:
        print(f"⚠ XGBoost model not found at {xgb_path}")

    # MPNN
    mpnn_path = os.environ.get("MPNN_PATH", "models/mpnn_solubility_best.pt")
    if os.path.exists(mpnn_path):
        try:
            from mpnn_model import MPNNSolubility
            MPNN_MODEL = MPNNSolubility.load(mpnn_path, device="cpu")
            print(f"✅ MPNN loaded: {mpnn_path}")
        except Exception as e:
            print(f"⚠ MPNN failed: {e}")
    else:
        print(f"⚠ MPNN model not found at {mpnn_path}")


# ── HELPERS ───────────────────────────────────────────────────────────────────
def compute_features(smiles: str):
    """298 features — exact pipeline from Water-Solubility.py"""
    from solubility_api import compute_all_features
    return compute_all_features(smiles)

def classify(logS: float):
    if logS > 0:   return "highly_soluble",        "Highly Soluble"
    if logS > -2:  return "soluble",               "Soluble"
    if logS > -4:  return "slightly_soluble",      "Slightly Soluble"
    return               "practically_insoluble",  "Practically Insoluble"

def domain_check(smiles: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        mw  = Descriptors.MolWt(mol) if mol else 999
        return {
            "in_domain": mw < 600,
            "domain_score": max(0, min(100, int(100*(1-max(0,mw-100)/700)))),
            "molecular_weight": round(mw, 2),
            "domain_note": f"MW={mw:.1f} — {'below' if mw<600 else 'above'} 600 Da threshold",
        }
    except:
        return {"in_domain": True, "domain_score": 50, "molecular_weight": None, "domain_note": ""}


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
class SMILESRequest(BaseModel):
    smiles: str

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "xgboost_loaded": XGB_MODEL is not None,
        "mpnn_loaded":    MPNN_MODEL is not None,
        "models": {
            "xgboost": {"features": 298, "file": "xgboost_model_298_4045.json"},
            "mpnn": {"layers": 4, "node_dim": 128, "file": "models/mpnn_solubility_best.pt"},
        }
    }

@app.post("/predict")
async def predict_xgb(req: SMILESRequest):
    """XGBoost prediction — 298 features."""
    if XGB_MODEL is None:
        raise HTTPException(503, "XGBoost model not loaded")
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(req.smiles)
        if mol is None:
            raise HTTPException(400, "Invalid SMILES string")
        feats = compute_features(req.smiles)
        logS  = float(XGB_MODEL.predict(feats)[0])
        mw    = Descriptors.MolWt(mol)
        cls, lbl = classify(logS)
        dom  = domain_check(req.smiles)
        return {
            "smiles": req.smiles,
            "logS": round(logS, 3),
            "mol_per_liter": round(10**logS, 5),
            "gram_per_liter": round((10**logS)*mw, 4),
            "molecular_weight": round(mw, 2),
            "solubility_class": cls,
            "solubility_label": lbl,
            **dom,
            "model": "xgboost_298_features",
            "n_features": feats.shape[1],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict_mpnn")
async def predict_mpnn(req: SMILESRequest):
    """MPNN prediction — molecular graph."""
    if MPNN_MODEL is None:
        raise HTTPException(503, "MPNN model not loaded. Run train_mpnn.py first.")
    try:
        logS = MPNN_MODEL.predict_smiles(req.smiles)
        if logS is None:
            raise HTTPException(400, "Invalid SMILES string")
        cls, lbl = classify(logS)
        dom = domain_check(req.smiles)
        return {
            "smiles": req.smiles,
            "logS": logS,
            "mol_per_liter": round(10**logS, 5),
            "solubility_class": cls,
            "solubility_label": lbl,
            **dom,
            "model": "mpnn_4layer_graph",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict_both")
async def predict_both(req: SMILESRequest):
    """Both models in one call."""
    results = {}
    try:
        xgb_res = await predict_xgb(req)
        results["xgboost"] = xgb_res
    except Exception as e:
        results["xgboost"] = {"error": str(e)}
    try:
        mpnn_res = await predict_mpnn(req)
        results["mpnn"] = mpnn_res
    except Exception as e:
        results["mpnn"] = {"error": str(e)}
    # Compute ensemble if both succeeded
    if "error" not in results.get("xgboost",{}) and "error" not in results.get("mpnn",{}):
        ens = (results["xgboost"]["logS"] + results["mpnn"]["logS"]) / 2
        results["ensemble"] = {
            "logS": round(ens, 3),
            "mol_per_liter": round(10**ens, 5),
            "solubility_label": classify(ens)[1],
            "agreement": round(abs(results["xgboost"]["logS"] - results["mpnn"]["logS"]), 3),
        }
    return results

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
