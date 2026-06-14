# Methodology

## Problem Definition

**Retrosynthesis prediction**: Given a target product molecule, predict the reactants required to synthesise it. This is framed as a sequence-to-sequence problem: product SMILES → reactant SMILES.

## Dataset: USPTO-50K

50,000 reactions from US patent literature (Liu et al. 2017).
- Train: 40,000 | Val: 5,000 | Test: 5,000
- 10 reaction classes
- Reagents removed
- Reaction class token `<RX_N>` prepended to product SMILES

## Model

Transformer encoder-decoder (OpenNMT-py):
- Embedding: 256 | Heads: 8 | Layers: 6 | FFN: 2048 | Steps: 100K

## SMILES Augmentation

Molecules have many valid SMILES representations. Randomising atom traversal order generates new training examples. This is the key contribution of the kheyer repo:

| Factor | Pairs | Top-1 | Top-5 |
|---|---|---|---|
| 1× (none) | 40K | 53.7% | 71.2% |
| 4× | 160K | 56.0% | 72.3% |
| **16×** | **640K** | **61.3%** | **74.2%** |
| 40× | 1.56M | 62.1% | 65.0% |

**16× is optimal**: 40× further boosts top-1 but causes the model to predict different SMILES representations of the same canonical molecule, artificially reducing top-k diversity.

## Canonicalisation Deduplication

Standard beam search wastes slots on SMILES variants. Our improved beam search:
1. Generates beam_size=10 hypotheses
2. Canonicalises each using RDKit
3. Removes duplicates
4. Returns top-k chemically unique predictions

This recovers top-k accuracy that 40× augmentation loses.

## Evaluation

Standard top-k accuracy on Liu et al. test set:
- Canonicalise predictions before comparison
- Match against canonical target SMILES
- Report top-1, 3, 5, 10 accuracy

## References

1. Liu et al. (2017). ACS Central Science. doi:10.1021/acscentsci.7b00303
2. Lin et al. (2019). arXiv:1906.02308
3. Bjerrum (2017). SMILES enumeration. arXiv:1703.07076
4. kheyer/Retrosynthesis-Prediction (GitHub)
