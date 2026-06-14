#!/bin/bash
# Preprocess USPTO-50K for retrosynthesis training
echo "Preprocessing USPTO-50K..."
python src/data/preprocess.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --max_len 512
echo "Done. Processed files in data/processed/"
