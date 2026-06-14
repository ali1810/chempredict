"""
preprocess.py — USPTO-50K Preprocessing Pipeline
=================================================
Tokenises SMILES, removes reagents, adds reaction class tokens.

Input format (USPTO):  reactants>reagents>products
Model input:           <RX_N> product_smiles (tokenised)
Model target:          reactant_smiles (tokenised)

Author: Dr. Mushtaq Ali · KIT
"""
import os, re, json, argparse
import pandas as pd
from typing import List, Tuple, Optional
from tqdm import tqdm

try:
    from rdkit import Chem
    RDK = True
except ImportError:
    RDK = False

# 10 USPTO reaction classes (Liu et al. 2017)
REACTION_CLASSES = {
    1: "heteroatom_alkylation_arylation",
    2: "acylation",
    3: "cc_bond_formation",
    4: "heterocycle_formation",
    5: "protections",
    6: "deprotections",
    7: "reductions",
    8: "oxidations",
    9: "fgi",
    10: "fga",
}

# Regex tokeniser — handles multi-char atoms, brackets, ring closures
SMI_RE = re.compile(r"(\%\d{2}|Br|Cl|Si|Se|@@|@|\[.*?\]|[BCNOPSFIbcnosp#=\-+/\\()\d\.])")


def tokenise(smiles: str) -> List[str]:
    """Tokenise SMILES at atom/bond level."""
    tokens = SMI_RE.findall(smiles)
    return tokens if tokens else list(smiles)


def canonicalise(smiles: str) -> Optional[str]:
    """Return canonical SMILES via RDKit. None if invalid."""
    if not RDK: return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


def parse_rxn(rxn: str) -> Tuple[str, str, str]:
    """Parse USPTO reaction SMILES → (reactants, reagents, products)."""
    parts = rxn.strip().split(">")
    if len(parts) != 3:
        raise ValueError(f"Bad reaction format: {rxn}")
    return parts[0], parts[1], parts[2]


def process_file(csv_path: str, out_dir: str, split: str,
                 canonicalise_smiles: bool = True,
                 max_len: int = 512):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    src_lines, tgt_lines = [], []
    stats = {"total": len(df), "valid": 0, "skipped": 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        try:
            rxn = str(row.get("rxn_smiles", row.get("reaction_smiles", "")))
            cls = int(row.get("reaction_class", row.get("class", 1)))
            reactants, _, products = parse_rxn(rxn)

            if canonicalise_smiles:
                reactants = canonicalise(reactants) or reactants
                products  = canonicalise(products)  or products

            src_tok = tokenise(products)
            tgt_tok = tokenise(reactants)

            if not src_tok or not tgt_tok: continue
            if len(src_tok) > max_len or len(tgt_tok) > max_len:
                stats["skipped"] += 1; continue

            # Prepend reaction class token to product (model input)
            src_lines.append(f"<RX_{cls}> " + " ".join(src_tok))
            tgt_lines.append(" ".join(tgt_tok))
            stats["valid"] += 1

        except Exception:
            stats["skipped"] += 1

    with open(f"{out_dir}/src-{split}.txt","w") as f: f.write("\n".join(src_lines))
    with open(f"{out_dir}/tgt-{split}.txt","w") as f: f.write("\n".join(tgt_lines))

    print(f"{split}: {stats['valid']}/{stats['total']} valid | {stats['skipped']} skipped")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--max_len", type=int, default=512)
    args = parser.parse_args()

    for split in ["train","val","test"]:
        fpath = f"{args.input_dir}/{split}.csv"
        if os.path.exists(fpath):
            process_file(fpath, args.output_dir, split, max_len=args.max_len)
