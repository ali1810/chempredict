"""
transformer.py — Transformer Seq2Seq for Retrosynthesis
========================================================
Architecture from kheyer/Retrosynthesis-Prediction:
  - Embedding dim: 256
  - Model dim:     256
  - FFN dim:      2048
  - Heads:           8
  - Layers:          6
  - Iterations: 100,000

Task: product SMILES (+ reaction class token) → reactant SMILES
This is a sequence-to-sequence NMT problem, same as translation.

Author: Dr. Mushtaq Ali · KIT
"""
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ── POSITIONAL ENCODING ────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── RETROSYNTHESIS TRANSFORMER ─────────────────────────────────────────────
class RetrosynthesisTransformer(nn.Module):
    """
    Transformer encoder-decoder for retrosynthesis prediction.

    Input:  product SMILES tokens (with <RX_N> class token prepended)
    Output: reactant SMILES tokens

    Trained on USPTO-50K with SMILES augmentation (16x recommended).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,       # From kheyer config
        nhead: int = 8,            # From kheyer config
        num_encoder_layers: int = 6,  # From kheyer config
        num_decoder_layers: int = 6,  # From kheyer config
        dim_feedforward: int = 2048,  # From kheyer config
        dropout: float = 0.1,
        max_len: int = 512,
        pad_idx: int = 0,
        share_embeddings: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        if share_embeddings:
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        """Encode product SMILES tokens."""
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None,
               tgt_key_padding_mask: Optional[torch.Tensor] = None):
        """Decode reactant SMILES tokens given encoder memory."""
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        return self.transformer.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.

        Parameters
        ----------
        src : (batch, src_len) — product token indices
        tgt : (batch, tgt_len) — reactant token indices (teacher-forced, shifted right)

        Returns
        -------
        logits : (batch, tgt_len, tgt_vocab_size)
        """
        src_pad_mask = (src == self.pad_idx)
        tgt_pad_mask = (tgt == self.pad_idx)
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=src.device)

        memory = self.encode(src, src_key_padding_mask=src_pad_mask)
        out = self.decode(tgt, memory,
                          tgt_mask=tgt_mask,
                          memory_key_padding_mask=src_pad_mask,
                          tgt_key_padding_mask=tgt_pad_mask)
        return self.output_proj(out)

    def save(self, path: str, vocab_src=None, vocab_tgt=None, config: dict = None):
        """Save model checkpoint with config."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": config or {
                "d_model": self.d_model,
                "src_vocab_size": self.src_embed.num_embeddings,
                "tgt_vocab_size": self.output_proj.out_features,
            },
            "vocab_src": vocab_src,
            "vocab_tgt": vocab_tgt,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RetrosynthesisTransformer":
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["config"]
        model = cls(**{k: v for k, v in cfg.items()})
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(device)


# ── CONVENIENCE WRAPPER ────────────────────────────────────────────────────
class RetrosynthesisModel:
    """
    High-level wrapper for inference.
    Handles tokenisation, beam search, and canonicalisation.
    """

    def __init__(self, model: RetrosynthesisTransformer,
                 vocab_src: dict, vocab_tgt: dict, device: str = "cpu"):
        self.model = model.to(device)
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.idx2tok = {v: k for k, v in vocab_tgt.items()}
        self.device = device
        self.model.eval()

    def predict(self, product_smiles: str, reaction_class: int = 1,
                beam_size: int = 10, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict top-k reactants for a given product SMILES.

        Parameters
        ----------
        product_smiles : str
            Target molecule as SMILES string
        reaction_class : int
            1-10, prepended as <RX_N> token
        beam_size : int
            Beam search width
        top_k : int
            Number of unique predictions to return

        Returns
        -------
        List of (reactant_smiles, score) tuples, ranked by score
        """
        from src.model.beam_search import beam_search
        from src.data.preprocess import tokenise, canonicalise

        # Tokenise input
        tokens = [f"<RX_{reaction_class}>"] + list(tokenise(product_smiles))
        indices = [self.vocab_src.get(t, self.vocab_src.get("<UNK>", 1)) for t in tokens]
        src = torch.tensor([indices], device=self.device)

        # Encode
        with torch.no_grad():
            src_pad_mask = (src == 0)
            memory = self.model.encode(src, src_key_padding_mask=src_pad_mask)

        # Beam search
        results = beam_search(
            model=self.model,
            memory=memory,
            src_pad_mask=src_pad_mask,
            vocab=self.vocab_tgt,
            idx2tok=self.idx2tok,
            beam_size=beam_size,
            max_len=200,
            device=self.device,
        )

        # Canonicalise and deduplicate (key fix for 40x augmentation problem)
        seen = {}
        for tokens_out, score in results:
            smiles = "".join(tokens_out)
            canon = canonicalise(smiles) or smiles
            if canon not in seen:
                seen[canon] = score

        return list(seen.items())[:top_k]

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "RetrosynthesisModel":
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = RetrosynthesisTransformer(**ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, ckpt["vocab_src"], ckpt["vocab_tgt"], device)
