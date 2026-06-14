"""
api.py — FastAPI Server for Model-Backed Predictions
=====================================================
When your trained model is available, this replaces the Claude AI
backend with your actual USPTO-50K trained Transformer.

Usage:
  pip install fastapi uvicorn
  uvicorn app.api:app --reload --port 8000
  # Then open http://localhost:8000/docs

Author: Dr. Mushtaq Ali · KIT
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI(
    title="RxnPredict API",
    description="Retrosynthesis prediction using Transformer seq2seq on USPTO-50K",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup (if checkpoint exists)
MODEL = None
CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "models/checkpoint_best.pt")

@app.on_event("startup")
async def load_model():
    global MODEL
    if os.path.exists(CHECKPOINT):
        try:
            from src.model.transformer import RetrosynthesisModel
            MODEL = RetrosynthesisModel.load(CHECKPOINT)
            print(f"Model loaded from {CHECKPOINT}")
        except Exception as e:
            print(f"Could not load model: {e}. Running in demo mode.")
    else:
        print(f"No checkpoint at {CHECKPOINT}. Running in demo mode.")


class PredictRequest(BaseModel):
    product_smiles: str
    reaction_class: int = 1         # 1-10 (USPTO classes)
    beam_size: int = 10
    top_k: int = 5
    deduplicate: bool = True        # Canonicalise & remove SMILES variant duplicates


class ReactantCandidate(BaseModel):
    rank: int
    smiles: str
    score: float
    confidence: str


class PredictResponse(BaseModel):
    product: str
    reaction_class: int
    model_input: str                # <RX_N> product (as fed to transformer)
    top_reactants: List[ReactantCandidate]
    in_applicability_domain: bool
    model_version: str = "16x_augmentation"


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Predict reactants for a given product SMILES.

    The model takes: <RX_{class}> product_smiles
    And outputs: reactant_smiles (dot-separated if multiple)
    """
    if req.reaction_class < 1 or req.reaction_class > 10:
        raise HTTPException(400, "reaction_class must be 1-10")

    model_input = f"<RX_{req.reaction_class}> {req.product_smiles}"

    if MODEL is not None:
        # Real model inference
        try:
            preds = MODEL.predict(
                product_smiles=req.product_smiles,
                reaction_class=req.reaction_class,
                beam_size=req.beam_size,
                top_k=req.top_k,
            )
            candidates = []
            for i, (smiles, score) in enumerate(preds):
                import math
                prob = math.exp(score)
                conf = "high" if prob > 0.7 else "medium" if prob > 0.4 else "low"
                candidates.append(ReactantCandidate(
                    rank=i+1, smiles=smiles,
                    score=round(prob, 3), confidence=conf
                ))
        except Exception as e:
            raise HTTPException(500, f"Prediction failed: {str(e)}")
    else:
        # Demo mode — return placeholder
        candidates = [
            ReactantCandidate(rank=1, smiles="[DEMO_MODE]",
                             score=0.0, confidence="low"),
        ]

    return PredictResponse(
        product=req.product_smiles,
        reaction_class=req.reaction_class,
        model_input=model_input,
        top_reactants=candidates,
        in_applicability_domain=MODEL is not None,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "checkpoint": CHECKPOINT,
        "dataset": "USPTO-50K",
        "architecture": "Transformer seq2seq, 256-dim, 8 heads, 6 layers",
        "augmentation": "16x SMILES randomisation",
        "top1_accuracy": "61.3%",
    }


@app.get("/reaction_classes")
async def reaction_classes():
    return {
        "classes": {
            1: "Heteroatom alkylation and arylation",
            2: "Acylation and related processes",
            3: "C-C bond formation",
            4: "Heterocycle formation",
            5: "Protections",
            6: "Deprotections",
            7: "Reductions",
            8: "Oxidations",
            9: "Functional group interconversion (FGI)",
            10: "Functional group addition (FGA)",
        },
        "note": "Reaction class token is prepended to product SMILES as <RX_N>",
        "source": "Liu et al. 2017, ACS Central Science, USPTO-50K benchmark",
    }
