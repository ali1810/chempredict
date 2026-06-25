"""
main.py — RxnPredict FastAPI Backend
Downloads model from HuggingFace Hub at startup,
loads OpenNMT model, runs real beam search inference.
"""
from __future__ import annotations
import os
import re
import math
import time
from pathlib import Path
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO       = "ali1810/retrosynthesis-opennmt"
MODEL_FILE    = "model_step_10000.pt"
SRC_VOCAB     = "USPTO.vocab.src"
TGT_VOCAB     = "USPTO.vocab.tgt"
MODEL_DIR     = Path("./model_cache")
BEAM_SIZE     = 5
N_BEST        = 5
MAX_LENGTH    = 300
GPU           = -1   # -1 = CPU

# ── SMILES tokenizer ──────────────────────────────────────────────────────────
SMILES_REGEX = re.compile(
    r"(\%\d{2}|Br|Cl|Si|Se|se|Na|Li|Ca|Mg|Fe|Cu|Zn|Ag|Au|Pt|"
    r"@@|@|\[|\]|=|#|-|\+|\\|\/|:|~|\.|\(|\)|\d|[A-Z]|[a-z])"
)

def tokenize(smiles: str) -> str:
    return " ".join(SMILES_REGEX.findall(smiles))

def detokenize(tokens: str) -> str:
    return "".join(tokens.split())

# ── Global model holder ───────────────────────────────────────────────────────
_translator = None

def download_model_files():
    """Download model and vocab files from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(exist_ok=True)
    files = [MODEL_FILE, SRC_VOCAB, TGT_VOCAB]

    for fname in files:
        dest = MODEL_DIR / fname
        if dest.exists():
            logger.info("Already cached: {}", fname)
            continue
        logger.info("Downloading {} from HF Hub...", fname)
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=fname,
            local_dir=str(MODEL_DIR),
        )
        logger.info("Downloaded: {} → {}", fname, path)


def load_model():
    """Load OpenNMT translator from cached model files."""
    global _translator

    import onmt.opts as opts
    from onmt.utils.parse import ArgumentParser
    from onmt.translate.translator import build_translator

    model_path = MODEL_DIR / MODEL_FILE
    src_vocab  = MODEL_DIR / SRC_VOCAB
    tgt_vocab  = MODEL_DIR / TGT_VOCAB

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    parser = ArgumentParser()
    opts.config_opts(parser)
    opts.translate_opts(parser)

    opt = parser.parse_args([
        "-model",      str(model_path),
        "-src",        "dummy",
        "-beam_size",  str(BEAM_SIZE),
        "-n_best",     str(N_BEST),
        "-max_length", str(MAX_LENGTH),
        "-gpu",        str(GPU),
        "-batch_size", "1",
        "-min_length", "1",
        "-verbose",
    ])

    _translator = build_translator(opt, report_score=False)
    logger.info("✅ OpenNMT model loaded | beam={} n_best={}", BEAM_SIZE, N_BEST)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RxnPredict API...")
    try:
        download_model_files()
        load_model()
    except Exception as e:
        logger.error("Startup error: {}", e)
        logger.warning("API will start but predictions will fail until model loads")
    yield
    logger.info("Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RxnPredict — Retrosynthesis API",
    version="1.0.0",
    description="OpenNMT retrosynthesis prediction using ali1810/retrosynthesis-opennmt",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    smiles: str
    reaction_class: int = 0       # 0 = unknown, 1–10 = USPTO class
    n_best: int = 5
    beam_size: int = 5


class Prediction(BaseModel):
    rank: int
    reactants: str
    score: float
    confidence: float
    is_valid: bool


class PredictResponse(BaseModel):
    product: str
    predictions: list[Prediction]
    top_prediction: str
    processing_time_ms: float
    model: str = HF_REPO


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    repo: str
    device: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def validate_smiles(smiles: str) -> bool:
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return bool(smiles and len(smiles) > 1)


def run_inference(smiles: str, reaction_class: int, n_best: int, beam_size: int) -> list[dict]:
    """Run OpenNMT beam search on tokenized SMILES."""
    if _translator is None:
        raise RuntimeError("Model not loaded")

    # Prepend reaction class token if provided (USPTO-50K format)
    if reaction_class > 0:
        input_smiles = f"<RX_{reaction_class}> {smiles}"
    else:
        input_smiles = smiles

    tokenized = tokenize(input_smiles)
    logger.debug("Input tokens: {}", tokenized)

    scores, predictions = _translator.translate(
        src=[tokenized],
        batch_size=1,
    )

    raw_scores = scores[0]
    raw_preds  = predictions[0]

    # Normalise scores to confidence
    float_scores = [float(s) for s in raw_scores]
    if float_scores:
        max_s  = max(float_scores)
        exps   = [math.exp(s - max_s) for s in float_scores]
        total  = sum(exps)
        confs  = [e / total for e in exps]
    else:
        confs = []

    results = []
    for rank, (pred, score, conf) in enumerate(
        zip(raw_preds, float_scores, confs), start=1
    ):
        if isinstance(pred, str):
            reactants = detokenize(pred)
        else:
            reactants = detokenize(" ".join(str(t) for t in pred))

        results.append({
            "rank":       rank,
            "reactants":  reactants,
            "score":      round(score, 4),
            "confidence": round(conf, 4),
            "is_valid":   validate_smiles(reactants),
        })
        logger.debug("Rank {}: {} | score={:.4f}", rank, reactants, score)

    return results


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    smiles = request.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES cannot be empty")

    if _translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet — try again in a moment")

    t0 = time.time()
    try:
        raw = run_inference(
            smiles,
            reaction_class=request.reaction_class,
            n_best=request.n_best,
            beam_size=request.beam_size,
        )
    except Exception as e:
        logger.error("Inference error: {}", e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    elapsed = round((time.time() - t0) * 1000, 1)

    predictions = [Prediction(**r) for r in raw]
    top = next((p.reactants for p in predictions if p.is_valid), None)
    if top is None and predictions:
        top = predictions[0].reactants

    return PredictResponse(
        product=smiles,
        predictions=predictions,
        top_prediction=top or "",
        processing_time_ms=elapsed,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if _translator is not None else "model_loading",
        model_loaded=_translator is not None,
        repo=HF_REPO,
        device="CPU" if GPU == -1 else f"GPU:{GPU}",
    )


@app.get("/examples")
async def examples():
    return {"examples": [
        {"name": "Aspirin",      "smiles": "CC(=O)Oc1ccccc1C(=O)O",       "class": 2},
        {"name": "Paracetamol",  "smiles": "CC(=O)Nc1ccc(O)cc1",           "class": 2},
        {"name": "Ibuprofen",    "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",   "class": 2},
        {"name": "Caffeine",     "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",   "class": 4},
        {"name": "Ethanol",      "smiles": "CCO",                           "class": 9},
    ]}
