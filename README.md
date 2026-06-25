# 🧪 ChemPredict — AI for Chemistry

<div align="center">

![ChemPredict Banner](docs/main_banner.png)

**Two production-ready AI tools for computational chemistry**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenNMT](https://img.shields.io/badge/OpenNMT-py-FF6B6B?style=for-the-badge)](https://opennmt.net)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-FFD21E?style=for-the-badge)](https://huggingface.co/ali1810)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![ACS JCIM](https://img.shields.io/badge/Published-ACS_JCIM_2025-blue?style=for-the-badge)](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02399)

| 🔬 Tool | 📊 Performance | 🌐 Live Demo |
|---|---|---|
| **AquaSolubility** | RMSE: 0.89 | [Try it →](https://ali1810.github.io/chempredict/solubility/) |
| **RxnPredict** | Top-1: 52%, Top-5: 68% | [Try it →](https://ali1810.github.io/chempredict/rxnpredict/) |

</div>

---

## 📦 Repository Structure

```
chempredict/
├── 🔵 solubility/          # Aqueous solubility prediction
│   ├── index.html          # Interactive demo frontend
│   ├── app.py              # Streamlit application
│   └── README.md           # Solubility-specific docs
│
├── 🟢 rxnpredict/          # Retrosynthesis prediction
│   ├── index.html          # Interactive demo frontend
│   ├── main.py             # FastAPI backend
│   └── README.md           # RxnPredict-specific docs
│
├── docs/                   # Images and documentation
└── index.html              # Portfolio homepage
```

---

## 🔵 Project 1 — AquaSolubility

![Solubility Banner](docs/solubility_banner.png)

> Predict the aqueous solubility of organic compounds using a hybrid
> ChemBERT + Message Passing Neural Network (MPNN) architecture.

### 📄 Publication

**Ali, M., Vanderheiden, S., Grathwol, C.W., Friederich, P., Jung, N., & Bräse, S. (2025).**
Advancing Aqueous Solubility Prediction: A Machine Learning Approach for Organic Compounds Using a Curated Dataset.
*Journal of Chemical Information and Modeling, ACS.*
→ [https://pubs.acs.org/doi/10.1021/acs.jcim.4c02399](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02399)

### ✨ Key Features

- **30,000+ curated molecular records** from PubChem
- **Hybrid architecture:** ChemBERT transformer + MPNN
- **Real-time inference** via Streamlit platform
- **Uncertainty-aware** predictions with confidence intervals
- **RMSE: 0.89 log mol/L** on held-out test set

### 🏗️ Architecture

```
Input SMILES
      │
      ├──→ ChemBERT Encoder ──→ Molecular Embeddings
      │                                │
      └──→ MPNN (Graph NN) ──→ Graph Embeddings
                                       │
                               Feature Fusion Layer
                                       │
                               Solubility Prediction
                               (log mol/L + confidence)
```

### 📊 Results

| Model | RMSE ↓ | R² ↑ | MAE ↓ |
|---|---|---|---|
| Random Forest | 1.24 | 0.71 | 0.94 |
| Standard MPNN | 1.05 | 0.78 | 0.82 |
| ChemBERT only | 0.98 | 0.81 | 0.76 |
| **ChemBERT + MPNN (ours)** | **0.89** | **0.85** | **0.68** |

### 🚀 Quick Start

```python
import requests

response = requests.post(
    "https://aqua-solubility-prediction.streamlit.app/predict",
    json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"}  # Aspirin
)
result = response.json()
print(f"Solubility: {result['solubility']} log mol/L")
print(f"Confidence: {result['confidence']}")
```

### 🌐 Live Demo

→ **[aqua-solubility-prediction.streamlit.app](https://aqua-solubility-prediction.streamlit.app)**

---

## 🟢 Project 2 — RxnPredict

![RxnPredict Banner](docs/rxnpredict_banner.png)

> Predict the reactants needed to synthesise a target molecule using
> a Transformer Seq2Seq model trained on 50,000 USPTO reactions.

### 🤗 Model on HuggingFace

→ **[huggingface.co/ali1810/retrosynthesis-opennmt](https://huggingface.co/ali1810/retrosynthesis-opennmt)**

API: **[ali1810-rxnpredict-api.hf.space/docs](https://ali1810-rxnpredict-api.hf.space/docs)**

### ✨ Key Features

- **USPTO-50K dataset** — 50,000 validated reactions
- **16× SMILES augmentation** — improves generalisation
- **Reaction class tokens** — 10 USPTO reaction classes
- **Beam search decoding** — top-5 candidate predictions
- **RDKit validation** — filters chemically invalid outputs
- **REST API** — production-ready FastAPI backend

### 🏗️ Architecture

```
Product SMILES
      │
  Tokenizer (character-level)
      │
  [<RX_2>] C C ( = O ) O c 1 c c c c c 1
      │
  Transformer Encoder (6 layers, 512d, 8 heads)
      │
  Transformer Decoder (6 layers) + Beam Search (k=5)
      │
  Top-5 Reactant Predictions
      │
  RDKit Validation + Confidence Scoring
```

### 📊 Results on USPTO-50K

| Metric | Score |
|---|---|
| **Top-1 Accuracy** | **52%** |
| **Top-3 Accuracy** | **65%** |
| **Top-5 Accuracy** | **68%** |
| Valid SMILES rate | 85% |
| Training steps | 8,000 |

### 🚀 Quick Start

```python
import requests

# Predict reactants for Aspirin
response = requests.post(
    "https://ali1810-rxnpredict-api.hf.space/predict",
    json={
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "reaction_class": 2,    # Acylation
        "n_best": 5
    }
)

data = response.json()
print(f"Product: {data['product']}")
print(f"\nTop predictions:")
for p in data["predictions"]:
    valid = "✅" if p["is_valid"] else "⚠️"
    print(f"  Rank {p['rank']}: {p['reactants']} "
          f"({p['confidence']*100:.1f}%) {valid}")
```

**Output:**
```
Product: CC(=O)Oc1ccccc1C(=O)O

Top predictions:
  Rank 1: CC(=O)O.Oc1ccccc1C(=O)O  (45.2%) ✅
  Rank 2: CC(=O)OC(=O)C.Oc1ccccc1  (28.1%) ✅
  Rank 3: CC(=O)Cl.Oc1ccccc1C(=O)O (15.3%) ✅
```

### 🌐 Live Demo

→ **[ali1810.github.io/chempredict/rxnpredict](https://ali1810.github.io/chempredict/rxnpredict/)**

---

## 🛠️ Installation & Local Setup

### Prerequisites

```bash
Python 3.10+
CUDA 11.8+ (optional, for GPU)
```

### Clone and Install

```bash
# Clone repository
git clone https://github.com/ali1810/chempredict.git
cd chempredict
```

### Run Solubility App

```bash
cd solubility
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

### Run RxnPredict API

```bash
cd rxnpredict
pip install -r requirements.txt

# Copy .env
cp .env.example .env

# Start API (downloads model from HF Hub automatically)
uvicorn main:app --reload
# → http://localhost:8000/docs
```

---

## 📈 Roadmap

- [x] Solubility prediction — published ACS JCIM 2025
- [x] Retrosynthesis model trained on USPTO-50K
- [x] Live demo deployed on HuggingFace Spaces
- [ ] Uncertainty quantification for retrosynthesis
- [ ] Reaction condition prediction (solvent, catalyst)
- [ ] USPTO-MIT (480k reactions) — larger dataset
- [ ] Attention visualization — explainable predictions
- [ ] Ensemble of 5 models — target 80% Top-1
- [ ] Beat SOTA (85%) with novel contributions

---

## 🔗 Links

| Resource | Link |
|---|---|
| 📄 ACS JCIM Paper | [pubs.acs.org/doi/10.1021/acs.jcim.4c02399](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02399) |
| 🤗 HuggingFace Models | [huggingface.co/ali1810](https://huggingface.co/ali1810) |
| 🌐 Solubility Demo | [aqua-solubility-prediction.streamlit.app](https://aqua-solubility-prediction.streamlit.app) |
| ⚗️ RxnPredict Demo | [ali1810.github.io/chempredict/rxnpredict](https://ali1810.github.io/chempredict/rxnpredict/) |
| 🔌 RxnPredict API | [ali1810-rxnpredict-api.hf.space/docs](https://ali1810-rxnpredict-api.hf.space/docs) |
| 👤 LinkedIn | [linkedin.com/in/mushtaq-ali](https://linkedin.com/in/mushtaq-ali) |
| 🐙 GitHub Profile | [ali1810.github.io/Profile](https://ali1810.github.io/Profile) |

---

## 👨‍🔬 Author

**Dr. Mushtaq Ali**
PhD in Artificial Intelligence · Karlsruhe Institute of Technology (KIT)
Sinsheim, Germany

*Computational scientist specialising in ML for molecular property prediction,
cheminformatics, and AI-driven drug discovery.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/mushtaq-ali)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-FFD21E?style=flat)](https://huggingface.co/ali1810)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/ali1810)

---

## 📝 Citation

If you use this work please cite:

```bibtex
@article{ali2025solubility,
  author    = {Ali, Mushtaq and Vanderheiden, S. and Grathwol, C.W.
               and Friederich, P. and Jung, N. and Br{\"a}se, S.},
  title     = {Advancing Aqueous Solubility Prediction: A Machine Learning
               Approach for Organic Compounds Using a Curated Dataset},
  journal   = {Journal of Chemical Information and Modeling},
  publisher = {ACS},
  year      = {2025},
  doi       = {10.1021/acs.jcim.4c02399}
}

@misc{ali2025retrosynthesis,
  author = {Ali, Mushtaq},
  title  = {RxnPredict: OpenNMT Retrosynthesis Prediction on USPTO-50K},
  year   = {2025},
  url    = {https://huggingface.co/ali1810/retrosynthesis-opennmt}
}
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  <sub>Built with ❤️ for the cheminformatics community</sub>
</div>
