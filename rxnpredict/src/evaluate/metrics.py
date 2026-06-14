"""
metrics.py — Top-K Accuracy Evaluation for Retrosynthesis
==========================================================
Standard evaluation protocol from Liu et al. (2017) on USPTO-50K.
Models are evaluated at Top-1, Top-3, Top-5, Top-10.

Key insight: Must canonicalise predictions before comparing.
Without canonicalisation, different SMILES of same molecule count
as different — artificially inflating top-k accuracy.

Author: Dr. Mushtaq Ali · KIT
"""
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

try:
    from rdkit import Chem
    RDK = True
except ImportError:
    RDK = False


def canonicalise(smiles: str) -> Optional[str]:
    """Return canonical SMILES. None if invalid."""
    if not RDK: return smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def canonicalise_reaction(smiles: str) -> Optional[str]:
    """Canonicalise all components of a multi-component SMILES (dot-separated)."""
    parts = smiles.split(".")
    canon_parts = []
    for p in parts:
        c = canonicalise(p)
        if c is None: return None
        canon_parts.append(c)
    return ".".join(sorted(canon_parts))  # Sort for order-invariant comparison


def topk_accuracy(
    predictions: List[List[str]],
    targets: List[str],
    k_values: Tuple[int, ...] = (1, 3, 5, 10),
    canonicalise_preds: bool = True,
) -> Dict[str, float]:
    """
    Compute top-k accuracy for retrosynthesis predictions.

    Parameters
    ----------
    predictions : List[List[str]]
        For each target, a list of predicted reactant SMILES (ranked best first)
    targets : List[str]
        Ground truth reactant SMILES for each target
    k_values : tuple
        Which k values to evaluate (standard: 1, 3, 5, 10)
    canonicalise_preds : bool
        If True, canonicalise and deduplicate predictions before scoring

    Returns
    -------
    dict : {"top1": 0.613, "top3": 0.709, "top5": 0.742, "top10": 0.764}
    """
    assert len(predictions) == len(targets), \
        f"Mismatch: {len(predictions)} predictions vs {len(targets)} targets"

    # Pre-canonicalise targets
    canon_targets = []
    for t in targets:
        c = canonicalise_reaction(t)
        canon_targets.append(c if c else t.strip())

    hits = defaultdict(int)
    max_k = max(k_values)

    for preds, target in zip(predictions, canon_targets):
        # Canonicalise predictions and deduplicate
        canon_preds = []
        seen = set()
        for p in preds:
            c = canonicalise_reaction(p) if canonicalise_preds else p.strip()
            if c and c not in seen:
                seen.add(c)
                canon_preds.append(c)
            if len(canon_preds) >= max_k:
                break

        # Check if target is in top-k
        for k in k_values:
            if target in canon_preds[:k]:
                hits[k] += 1

    n = len(targets)
    return {f"top{k}": round(hits[k] / n * 100, 1) for k in k_values}


def accuracy_by_class(
    predictions: List[List[str]],
    targets: List[str],
    reaction_classes: List[int],
    k: int = 10,
) -> Dict[int, float]:
    """
    Compute top-k accuracy broken down by reaction class.
    Reproduces Table from kheyer repo: per-class Top-10 accuracy.
    """
    class_preds = defaultdict(list)
    class_tgts  = defaultdict(list)

    for pred, tgt, cls in zip(predictions, targets, reaction_classes):
        class_preds[cls].append(pred)
        class_tgts[cls].append(tgt)

    results = {}
    for cls in sorted(class_preds.keys()):
        acc = topk_accuracy(class_preds[cls], class_tgts[cls], k_values=(k,))
        results[cls] = acc[f"top{k}"]

    return results


def diversity_analysis(
    predictions: List[List[str]],
    beam_size: int = 10,
) -> Dict[str, float]:
    """
    Analyse prediction diversity — the key 40x augmentation tradeoff.

    Measures average number of chemically unique predictions per input.
    Lower diversity = model is predicting SMILES variants, not unique molecules.

    From kheyer results:
    - No aug:  ~8.2 unique / 10 beams
    - 4x aug:  ~7.9 unique / 10 beams
    - 16x aug: ~7.1 unique / 10 beams
    - 40x aug: ~5.2 unique / 10 beams  ← big diversity drop
    """
    unique_counts = []
    for preds in predictions:
        seen = set()
        for p in preds[:beam_size]:
            c = canonicalise_reaction(p)
            if c: seen.add(c)
        unique_counts.append(len(seen))

    avg = sum(unique_counts) / len(unique_counts) if unique_counts else 0
    return {
        "beam_size": beam_size,
        "avg_unique_per_beam": round(avg, 2),
        "diversity_ratio": round(avg / beam_size, 3),
        "interpretation": (
            "Good diversity (>0.8)" if avg/beam_size > 0.8 else
            "Moderate — some SMILES variant duplicates" if avg/beam_size > 0.6 else
            "Low diversity — heavy SMILES variant duplication (typical of 40x augmentation)"
        )
    }


def print_results_table(results: Dict[str, float], model_name: str = "Our Model"):
    """Print results in format matching kheyer repo paper table."""
    print(f"\n{'Model':<25} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'Top-10':>8}")
    print("-" * 58)
    print(f"{'Liu et al. (LSTM)':<25} {'37.4':>8} {'52.4':>8} {'57.0':>8} {'61.7':>8}")
    print(f"{'Lin et al. (Transformer)':<25} {'54.6':>8} {'74.8':>8} {'80.2':>8} {'84.9':>8}")
    print(f"{'No Augmentation':<25} {'53.7':>8} {'67.7':>8} {'71.2':>8} {'73.9':>8}")
    print(f"{'16x Augmentation (kheyer)':<25} {'61.3':>8} {'70.9':>8} {'74.2':>8} {'76.4':>8}")
    print("-" * 58)
    row = "  ".join(f"{results.get(f'top{k}', '-'):>6}" for k in [1,3,5,10])
    print(f"{model_name:<25} {row}")
