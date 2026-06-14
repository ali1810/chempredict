# RxnPredict — Retrosynthesis & Reaction Prediction with Transformer + SMILES Augmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![RDKit](https://img.shields.io/badge/RDKit-2023-green)](https://rdkit.org)
[![Dataset](https://img.shields.io/badge/Dataset-USPTO--50K-purple)](https://pubs.acs.org/doi/10.1021/acscentsci.7b00303)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)

> **Transformer seq2seq retrosynthesis on USPTO-50K with SMILES augmentation**
> PhD Research — Karlsruhe Institute of Technology (KIT)
> Dr. Mushtaq Ali · ORCID: 0009-0007-6808-5783

---

## Overview

This repository implements end-to-end **AI-driven retrosynthesis prediction** — predicting reactants from a target product molecule. The system uses a **Transformer sequence-to-sequence model** trained on **USPTO-50K** with **SMILES-based data augmentation**.

```
Product  (Input):   O=C(O)c1ccc(Nc2nccc(-c3cccnc3)n2)cc1
         ↓ Transformer Seq2Seq (16× SMILES augmentation)
Reactants (Output): O=C(O)c1ccc(N)cc1 . Clc1nccc(-c2cccnc2)n1
```

---

## Results on USPTO-50K Benchmark

| Model | Top-1 | Top-3 | Top-5 | Top-10 |
|---|---|---|---|---|
| Liu et al. 2017 (LSTM) | 37.4% | 52.4% | 57.0% | 61.7% |
| Lin et al. 2019 (Transformer) | 54.6% | **74.8%** | **80.2%** | **84.9%** |
| **Ours — No Augmentation** | 53.7% | 67.7% | 71.2% | 73.9% |
| **Ours — 4× Augmentation** | 56.0% | 67.6% | 72.3% | 76.5% |
| **Ours — 16× Augmentation** | **61.3%** | 70.9% | 74.2% | 76.4% |
| **Ours — 40× Augmentation** | 62.1% | 64.1% | 65.0% | 66.4% |

**Key finding:** 16× augmentation is the sweet spot — boosts top-1 from 53.7% → 61.3% without hurting top-k. 40× further boosts top-1 but causes the model to predict different SMILES variants of the same molecule, reducing effective top-k diversity.

---

## Repository Structure

```
rxnpredict/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
│
├── src/
│   ├── data/
│   │   ├── preprocess.py      ← Tokenise USPTO-50K, add <RX_N> class tokens
│   │   ├── augmentation.py    ← SMILES randomisation, Nx dataset generation
│   │   └── dataset.py         ← PyTorch Dataset + DataLoader
│   │
│   ├── model/
│   │   ├── transformer.py     ← Transformer encoder-decoder seq2seq
│   │   ├── attention.py       ← Multi-head self + cross attention
│   │   └── beam_search.py     ← Beam search with canonicalisation filter
│   │
│   ├── train/
│   │   ├── trainer.py         ← Training loop, validation, checkpointing
│   │   └── config.py          ← Hyperparameter config (256-dim, 8-head, 6-layer)
│   │
│   └── evaluate/
│       ├── metrics.py         ← Top-k accuracy, canonicalisation deduplication
│       └── evaluate.py        ← Full evaluation pipeline
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_smiles_augmentation.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation_results.ipynb
│   └── 05_retrosynthesis_planning.ipynb
│
├── app/
│   ├── rxnpredict_app.html    ← Standalone browser demo (Claude AI backend)
│   └── api.py                 ← FastAPI server for model-backed predictions
│
├── scripts/
│   ├── preprocess.sh          ← One-command preprocessing
│   ├── train.sh               ← One-command training
│   └── evaluate.sh            ← One-command evaluation
│
├── data/
│   └── README.md              ← Data download instructions
│
└── docs/
    └── methodology.md         ← Full methodology write-up
```

---

## Quick Start

```bash
git clone https://github.com/ali1810/rxnpredict.git
cd rxnpredict
pip install -r requirements.txt
```

### 1 — Preprocess USPTO-50K
```bash
bash scripts/preprocess.sh
```

### 2 — Augment Training Data (16×)
```bash
python src/data/augmentation.py \
  --src data/processed/src-train.txt \
  --tgt data/processed/tgt-train.txt \
  --output data/augmented/16x \
  --factor 16
```

### 3 — Train
```bash
bash scripts/train.sh
```

### 4 — Evaluate
```bash
bash scripts/evaluate.sh
```

### 5 — Single Prediction
```python
from src.model.transformer import RetrosynthesisModel

model = RetrosynthesisModel.load('models/checkpoint_best.pt')
product = 'O=C(O)c1ccc(Nc2nccc(-c3cccnc3)n2)cc1'
preds = model.predict(product, reaction_class=2, beam_size=10, top_k=5)
for rank, (smi, score) in enumerate(preds, 1):
    print(f"Top-{rank}: {smi}  [{score:.3f}]")
```

---

## Methodology

### Dataset
USPTO-50K: 50,000 reactions from US patents. Train/val/test = 40k/5k/5k.
Reagents removed. 10 reaction classes, class token prepended to input.

```
Input:  <RX_6> C/C=C/c1cc(C(=O)O)c(F)cc1OCC12CC3CC(CC(C3)C1)C2
Output: C/C=C/c1cc(C(=O)OC(C)(C)C)c(F)cc1OCC12CC3CC(CC(C3)C1)C2
```

### Model
Transformer seq2seq, 256-dim, 8 heads, 6 layers, 2048 FF, 100K iterations.

### SMILES Augmentation
Molecules have many valid SMILES. Randomising atom traversal generates new representations:
```
CCO  →  OCC  |  C(O)C  |  C(C)O  ...
```
4× = 160K pairs · 16× = 640K pairs · 40× = 1.56M pairs

### Key Tradeoff
Higher augmentation → better top-1 but worse top-k (model predicts SMILES variants not unique molecules). 16× is optimal.

---

## Live Demo

Open `app/rxnpredict_app.html` in any browser — no installation needed.

---

## Citation

```bibtex
@article{ali2025solubility,
  title={Advancing Aqueous Solubility Prediction},
  author={Ali, Mushtaq and others},
  journal={ACS JCIM},
  year={2025},
  doi={10.1021/acs.jcim.4c02399}
}
```

## Author
**Dr. Mushtaq Ali** · KIT · [github.com/ali1810](https://github.com/ali1810) · info@dream2europe.com
