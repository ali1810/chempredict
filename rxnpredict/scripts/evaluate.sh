#!/bin/bash
# Evaluate model on USPTO-50K test set
CHECKPOINT=${1:-models/checkpoint_best.pt}
echo "Evaluating: $CHECKPOINT"
python src/evaluate/evaluate.py \
  --checkpoint $CHECKPOINT \
  --src data/processed/src-test.txt \
  --tgt data/processed/tgt-test.txt \
  --beam_size 10 \
  --output results/predictions.txt
