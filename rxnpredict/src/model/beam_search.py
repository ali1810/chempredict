"""
beam_search.py — Beam Search with Canonicalisation Deduplication
================================================================
Key innovation: After standard beam search, we canonicalise each
hypothesis using RDKit and deduplicate. This fixes the top-k diversity
problem observed with 40x SMILES augmentation.

Without dedup: model predicts "CCO", "OCC", "C(O)C" — same molecule, 3 slots
With dedup:    model returns 3 genuinely different reactant sets

Author: Dr. Mushtaq Ali · KIT
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

try:
    from rdkit import Chem
    RDK = True
except ImportError:
    RDK = False


def canonical_smiles(smiles: str) -> Optional[str]:
    if not RDK: return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def beam_search(
    model,
    memory: torch.Tensor,
    src_pad_mask: torch.Tensor,
    vocab: Dict[str, int],
    idx2tok: Dict[int, str],
    beam_size: int = 10,
    max_len: int = 200,
    device: str = "cpu",
    deduplicate: bool = True,
) -> List[Tuple[List[str], float]]:
    """
    Beam search decoder with optional canonicalisation deduplication.

    Parameters
    ----------
    model : RetrosynthesisTransformer
    memory : (1, src_len, d_model) — encoder output
    vocab : token → index mapping
    idx2tok : index → token mapping
    beam_size : number of beams
    deduplicate : if True, canonicalise and remove duplicate SMILES predictions

    Returns
    -------
    List of (token_list, log_prob_score) sorted best first
    """
    sos_idx = vocab.get("<SOS>", 1)
    eos_idx = vocab.get("<EOS>", 2)
    pad_idx = vocab.get("<PAD>", 0)

    # Initialise beams: [(tokens, log_prob)]
    beams = [([sos_idx], 0.0)]
    completed = []

    for step in range(max_len):
        if not beams:
            break

        # Expand memory for all active beams
        n_beams = len(beams)
        mem = memory.expand(n_beams, -1, -1)
        mem_mask = src_pad_mask.expand(n_beams, -1)

        # Build current decoder input
        max_tgt_len = max(len(b[0]) for b in beams)
        tgt_tensor = torch.full((n_beams, max_tgt_len), pad_idx,
                                dtype=torch.long, device=device)
        for i, (toks, _) in enumerate(beams):
            tgt_tensor[i, :len(toks)] = torch.tensor(toks, device=device)

        with torch.no_grad():
            import math
            tgt_emb = model.pos_enc(model.tgt_embed(tgt_tensor) * math.sqrt(model.d_model))
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                max_tgt_len, device=device)
            dec_out = model.transformer.decoder(
                tgt_emb, mem,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=mem_mask,
            )
            # Take last token position
            logits = model.output_proj(dec_out[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)

        # Expand beams
        new_beams = []
        topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)

        for beam_idx, (tokens, beam_score) in enumerate(beams):
            for k in range(beam_size):
                next_tok = topk_indices[beam_idx, k].item()
                next_score = beam_score + topk_log_probs[beam_idx, k].item()
                new_tokens = tokens + [next_tok]

                if next_tok == eos_idx:
                    completed.append((new_tokens[1:-1], next_score))  # strip SOS/EOS
                else:
                    new_beams.append((new_tokens, next_score))

        # Keep top beam_size beams by score
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

        if len(completed) >= beam_size:
            break

    # Include any incomplete beams
    for tokens, score in beams:
        completed.append((tokens[1:], score))

    # Convert indices to tokens
    results = []
    for indices, score in completed:
        tokens = [idx2tok.get(i, "") for i in indices]
        results.append((tokens, score))

    results.sort(key=lambda x: x[1], reverse=True)

    if not deduplicate:
        return results[:beam_size]

    # ── CANONICALISATION DEDUPLICATION ────────────────────────────────────
    # This is the key fix for the 40x augmentation top-k diversity problem
    seen_canonical = {}
    for tokens, score in results:
        smiles = "".join(tokens)
        canon = canonical_smiles(smiles) or smiles
        if canon not in seen_canonical:
            seen_canonical[canon] = (tokens, score)

    deduped = list(seen_canonical.values())
    deduped.sort(key=lambda x: x[1], reverse=True)
    return deduped[:beam_size]
