# Data

## Download USPTO-50K

The USPTO-50K dataset (50,000 reactions, 10 classes) is available from:

1. **Original paper**: Liu et al. (2017) ACS Central Science
   https://pubs.acs.org/doi/10.1021/acscentsci.7b00303

2. **GitHub (preprocessed)**: 
   https://github.com/connorcoley/retrosim/tree/master/data

3. **Augmented datasets** (from kheyer repo):
   https://www.dropbox.com/s/ze4bdif8sqjx5jx/Retrosynthesis%20Data.zip

## Directory Structure After Download

```
data/
├── raw/
│   ├── train.csv      # 40,000 reactions
│   ├── val.csv        # 5,000 reactions
│   └── test.csv       # 5,000 reactions
│
├── processed/         # After running scripts/preprocess.sh
│   ├── src-train.txt
│   ├── tgt-train.txt
│   ├── src-val.txt
│   ├── tgt-val.txt
│   ├── src-test.txt
│   └── tgt-test.txt
│
└── augmented/         # After augmentation
    ├── 4x/
    ├── 16x/           # Recommended
    └── 40x/
```

## CSV Format

Each CSV requires columns: `rxn_smiles`, `reaction_class`

```
rxn_smiles,reaction_class
"CCO.CC(=O)O>>[H+]>CC(=O)OCC",1
```
