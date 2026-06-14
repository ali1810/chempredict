"""
augmentation.py — SMILES-Based Data Augmentation
=================================================
Core insight from kheyer/Retrosynthesis-Prediction:
  - 4x  → top-1: 56.0%, top-5: 72.3%
  - 16x → top-1: 61.3%, top-5: 74.2%  ← SWEET SPOT
  - 40x → top-1: 62.1%, top-5: 65.0%  ← top-k diversity drops

Why 40x hurts top-k: model learns to predict different SMILES variants
of the same canonical molecule, wasting beam search slots.

Author: Dr. Mushtaq Ali · KIT
"""
import os, random, argparse
from typing import List, Optional, Tuple
from tqdm import tqdm

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("RDKit required: pip install rdkit")


def randomise_smiles(smiles: str) -> Optional[str]:
    """
    Generate one random non-canonical SMILES for a molecule.
    Shuffles atom traversal order — same molecule, different string.

    CCO  →  OCC  or  C(O)C  or  C(C)O  ...
    """
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None: return None
        n = mol.GetNumAtoms()
        if n == 0: return smiles
        order = list(range(n))
        random.shuffle(order)
        return Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
    except Exception:
        return None


def canonical_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


def augment_pair(src_line: str, tgt_line: str, factor: int) -> List[Tuple[str,str]]:
    """
    Generate `factor` augmented versions of one src/tgt pair.

    src_line: tokenised source e.g. "<RX_2> C c 1 c c c c c 1"
    tgt_line: tokenised target e.g. "C c 1 c c c c c 1 . N"
    """
    # Extract class token
    rxn_token = ""
    src_smiles = src_line
    if src_line.startswith("<RX_"):
        idx = src_line.index(">") + 2
        rxn_token = src_line[:idx]
        src_smiles = src_line[idx:]

    # Detokenise (remove spaces)
    src_smi = src_smiles.replace(" ", "")
    tgt_smi = tgt_line.replace(" ", "")

    pairs = []
    for _ in range(factor):
        # Augment each dot-separated component independently
        aug_src = ".".join(randomise_smiles(c) or c for c in src_smi.split("."))
        aug_tgt = ".".join(randomise_smiles(c) or c for c in tgt_smi.split("."))

        # Retokenise character by character (simple split)
        pairs.append((
            rxn_token + " ".join(list(aug_src)),
            " ".join(list(aug_tgt))
        ))
    return pairs


def augment_dataset(src_file: str, tgt_file: str, out_dir: str,
                    factor: int = 16, seed: int = 42, shuffle: bool = True):
    """
    Augment full training dataset.

    Parameters
    ----------
    factor : int
        4, 16, or 40 recommended. 16 is optimal.
    """
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    with open(src_file) as f: src_lines = [l.strip() for l in f if l.strip()]
    with open(tgt_file) as f: tgt_lines = [l.strip() for l in f if l.strip()]

    assert len(src_lines) == len(tgt_lines)
    print(f"Augmenting {len(src_lines):,} pairs × {factor} → ~{len(src_lines)*factor:,}")

    aug_src, aug_tgt = [], []
    for s, t in tqdm(zip(src_lines, tgt_lines), total=len(src_lines)):
        try:
            for as_, at_ in augment_pair(s, t, factor):
                aug_src.append(as_)
                aug_tgt.append(at_)
        except Exception:
            aug_src.append(s); aug_tgt.append(t)

    if shuffle:
        pairs = list(zip(aug_src, aug_tgt))
        random.shuffle(pairs)
        aug_src, aug_tgt = zip(*pairs)

    # Detect split from filename
    split = "train"
    for sp in ["val","test"]:
        if sp in src_file: split = sp

    with open(f"{out_dir}/src-{split}.txt","w") as f: f.write("\n".join(aug_src))
    with open(f"{out_dir}/tgt-{split}.txt","w") as f: f.write("\n".join(aug_tgt))
    print(f"Written to {out_dir}: {len(aug_src):,} pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMILES augmentation for retrosynthesis")
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--factor", type=int, default=16,
                        choices=[4,16,40],
                        help="4=160K, 16=640K (recommended), 40=1.56M")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.factor == 40:
        print("Warning: 40x improves top-1 but reduces top-k diversity.")
        print("         16x recommended for best overall performance.")

    augment_dataset(args.src, args.tgt, args.output, args.factor, args.seed)
