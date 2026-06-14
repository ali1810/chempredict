"""
config.py — Model & Training Configuration
==========================================
Exact hyperparameters from kheyer/Retrosynthesis-Prediction

Author: Dr. Mushtaq Ali · KIT
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer architecture — matches kheyer repo exactly."""
    d_model: int = 256          # Embedding + model dimension
    nhead: int = 8              # Attention heads
    num_encoder_layers: int = 6 # Encoder layers
    num_decoder_layers: int = 6 # Decoder layers
    dim_feedforward: int = 2048 # FFN dimension
    dropout: float = 0.1
    max_len: int = 512
    share_embeddings: bool = False


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Data
    data_dir: str = "data/augmented/16x"
    output_dir: str = "models"
    augmentation_factor: int = 16  # 4, 16, or 40

    # Training
    batch_size: int = 64
    max_epochs: int = 50
    max_iterations: int = 100_000   # From kheyer config
    learning_rate: float = 1e-4
    warmup_steps: int = 8_000       # Transformer warmup
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    seed: int = 42

    # Evaluation
    eval_every: int = 2_000
    save_every: int = 5_000
    beam_size: int = 10
    top_k: int = (1, 3, 5, 10)

    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    fp16: bool = False


@dataclass
class InferenceConfig:
    """Inference configuration."""
    checkpoint: str = "models/checkpoint_best.pt"
    beam_size: int = 10
    top_k: int = 5
    device: str = "cpu"
    deduplicate: bool = True    # Canonicalise & deduplicate beam outputs


# ── OPENNMT YAML CONFIG ────────────────────────────────────────────────────
OPENNMT_CONFIG = """
# model_config.yml
# Equivalent to kheyer/Retrosynthesis-Prediction/model_details/model_config.yml
# Use with: onmt_train -config model_config.yml

# Data paths (update after preprocessing)
data:
  corpus_1:
    path_src: data/augmented/16x/src-train.txt
    path_tgt: data/augmented/16x/tgt-train.txt
  valid:
    path_src: data/processed/src-val.txt
    path_tgt: data/processed/tgt-val.txt

# Vocabulary
src_vocab: data/vocab/vocab.src
tgt_vocab: data/vocab/vocab.tgt
share_vocab: true
src_vocab_size: 500
tgt_vocab_size: 500

# Transformer architecture (matches kheyer repo)
model_dim: 256
encoder_type: transformer
decoder_type: transformer
enc_layers: 6
dec_layers: 6
heads: 8
transformer_ff: 2048
dropout: 0.1
attention_dropout: 0.1

# Training
train_steps: 100000
valid_steps: 5000
save_checkpoint_steps: 5000
keep_checkpoint: 10
seed: 42
report_every: 100

batch_type: tokens
batch_size: 4096
max_generator_batches: 32
normalization: tokens
accum_count: 4

optim: adam
adam_beta1: 0.9
adam_beta2: 0.998
decay_method: noam
warmup_steps: 8000
learning_rate: 2.0
max_grad_norm: 0.0
label_smoothing: 0.0

save_model: models/model
log_file: logs/train.log
"""
