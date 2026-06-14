#!/bin/bash
# Train retrosynthesis transformer (16x augmentation recommended)
FACTOR=${1:-16}
echo "Training with ${FACTOR}x augmentation..."

# Step 1: Augment
python src/data/augmentation.py \
  --src data/processed/src-train.txt \
  --tgt data/processed/tgt-train.txt \
  --output data/augmented/${FACTOR}x \
  --factor $FACTOR

# Step 2: Train (using OpenNMT if available, else PyTorch trainer)
if command -v onmt_train &> /dev/null; then
  echo "Using OpenNMT-py..."
  sed "s|16x|${FACTOR}x|g" src/train/config.py | grep -A 50 "OPENNMT_CONFIG" | \
    tail -n +2 | head -n -1 > /tmp/model_config.yml
  onmt_train -config /tmp/model_config.yml
else
  echo "Using PyTorch trainer..."
  python -c "
from src.train.trainer import Trainer
from src.train.config import TrainConfig
cfg = TrainConfig(augmentation_factor=$FACTOR)
print('Configure your DataLoader and run Trainer.train()')
print('See notebooks/03_model_training.ipynb for full walkthrough')
"
fi
