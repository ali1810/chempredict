"""
mpnn_model.py
=============
Message Passing Neural Network (MPNN) for water solubility prediction.
Companion to XGBoost model (xgboost_model_298_4045.json).

Architecture:
  SMILES → Molecular Graph → MPNN → logS

Node features (per atom, 72 dimensions):
  - Atomic number (one-hot, 44)
  - Degree (one-hot, 11)
  - Formal charge (one-hot, 6)
  - Hybridisation (one-hot, 5)
  - Aromaticity (1)
  - Hydrogen count (one-hot, 5)

Edge features (per bond, 12 dimensions):
  - Bond type (one-hot, 4)
  - Ring membership (1)
  - Conjugation (1)
  - Stereo (one-hot, 6)

Training:
  Dataset:   final_unique_train.csv / final_unique_test.csv
  Target:    logS (log10 molar solubility)
  Optimizer: Adam, lr=1e-3, weight decay=1e-5
  Epochs:    100 with early stopping (patience=15)
  Scheduler: ReduceLROnPlateau

Author: Dr. Mushtaq Ali · KIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import os
import json
from typing import Optional, List, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDK = True
except ImportError:
    RDK = False

# ── ATOM FEATURE DIMENSIONS ─────────────────────────────────────────────────
ATOM_TYPES = ['C','N','O','S','F','Si','P','Cl','Br','I','B','Se',
               'Na','K','Ca','Mg','Al','As','Ge','Te','other']
DEGREES     = list(range(11))
CHARGES     = [-2,-1,0,1,2,5]
HYBS        = ['S','SP','SP2','SP3','SP3D','SP3D2','OTHER']
H_COUNTS    = [0,1,2,3,4]

ATOM_DIM    = len(ATOM_TYPES) + len(DEGREES) + len(CHARGES) + len(HYBS) + 1 + len(H_COUNTS)
# = 21 + 11 + 6 + 7 + 1 + 5 = 51

BOND_TYPES  = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
               Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] if RDK else []
BOND_DIM    = 4 + 1 + 1   # bond_type + ring + conjugated = 6


# ── FEATURISATION ────────────────────────────────────────────────────────────
def one_hot(val, choices, encode_other=True):
    enc = [1 if val == c else 0 for c in choices]
    if encode_other:
        enc.append(1 if val not in choices else 0)
    return enc

def atom_features(atom) -> List[float]:
    sym = atom.GetSymbol()
    feats  = one_hot(sym, ATOM_TYPES[:-1])          # 21
    feats += one_hot(atom.GetDegree(), DEGREES)      # 11
    feats += one_hot(atom.GetFormalCharge(), CHARGES)# 6
    hyb = str(atom.GetHybridization()).split('.')[-1]
    feats += one_hot(hyb, HYBS[:-1])                # 7
    feats += [1 if atom.GetIsAromatic() else 0]     # 1
    feats += one_hot(int(atom.GetTotalNumHs()), H_COUNTS) # 5
    return feats  # 51 total

def bond_features(bond) -> List[float]:
    bt = bond.GetBondType()
    feats  = [1 if bt == t else 0 for t in BOND_TYPES]  # 4
    feats += [1 if bond.IsInRing() else 0]               # 1
    feats += [1 if bond.GetIsConjugated() else 0]        # 1
    return feats  # 6 total

def smiles_to_graph(smiles: str) -> Optional[dict]:
    """
    Convert SMILES to graph dictionary.
    Returns: {node_feats, edge_index, edge_feats, n_atoms}
    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_feats = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float
    )

    # Edge index + features (undirected → add both directions)
    rows, cols, e_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        rows += [i, j]; cols += [j, i]
        e_feats += [bf, bf]

    if not rows:
        # Single atom — no edges
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_feats = torch.zeros(0, BOND_DIM, dtype=torch.float)
    else:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_feats = torch.tensor(e_feats, dtype=torch.float)

    return {
        'node_feats': node_feats,
        'edge_index': edge_index,
        'edge_feats': edge_feats,
        'n_atoms': node_feats.size(0)
    }


# ── MPNN LAYERS ──────────────────────────────────────────────────────────────
class MPNNLayer(nn.Module):
    """
    One message passing step.
    For each node: aggregate neighbour messages → update node state.
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        # Message function: [h_i || h_j || e_ij] → message
        self.message_fn = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        # Update function: [h_i || agg_message] → h_i_new
        self.update_fn = nn.GRUCell(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h, edge_index, edge_feats):
        if edge_index.size(1) == 0:
            return h  # no edges, no update

        src, dst = edge_index[0], edge_index[1]

        # Compute messages
        msg_input = torch.cat([h[src], h[dst], edge_feats], dim=-1)
        messages   = self.message_fn(msg_input)

        # Aggregate (mean) per node
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        counts = torch.zeros(h.size(0), 1, device=h.device)
        counts.scatter_add_(0, dst.unsqueeze(-1),
                            torch.ones(src.size(0), 1, device=h.device))
        counts = counts.clamp(min=1)
        agg = agg / counts

        # GRU update
        h_new = self.update_fn(agg, h)
        return self.norm(h_new)


class MPNNSolubility(nn.Module):
    """
    MPNN model for solubility prediction.

    Architecture:
      Input projection → N × MPNN layers → Readout (mean+max) → MLP → logS

    Parameters:
      node_dim:    atom embedding dimension (default 128)
      edge_dim:    bond feature dimension (6)
      n_layers:    number of message passing steps (default 4)
      hidden_dim:  MLP hidden dimension (default 256)
      dropout:     dropout rate (default 0.1)
    """
    def __init__(
        self,
        node_in:    int = ATOM_DIM,    # 51
        edge_in:    int = BOND_DIM,    # 6
        node_dim:   int = 128,
        n_layers:   int = 4,
        hidden_dim: int = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.node_in   = node_in
        self.edge_in   = edge_in
        self.node_dim  = node_dim
        self.n_layers  = n_layers
        self.hidden_dim= hidden_dim

        # Input projection
        self.input_proj = nn.Linear(node_in, node_dim)

        # Message passing layers
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(node_dim, edge_in, hidden_dim)
            for _ in range(n_layers)
        ])

        # Readout MLP
        # Concatenate mean and max pooling → 2 * node_dim input
        self.readout = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_feats, edge_index, edge_feats, batch_idx=None):
        """
        Args:
            node_feats: (N_total_atoms, node_in)
            edge_index: (2, E_total)
            edge_feats: (E_total, edge_in)
            batch_idx:  (N_total_atoms,) — which molecule each atom belongs to

        Returns:
            logS: (batch_size, 1) or (1, 1) for single molecule
        """
        # Project atom features
        h = F.relu(self.input_proj(node_feats))

        # Message passing
        for layer in self.mpnn_layers:
            h = layer(h, edge_index, edge_feats)

        # Graph-level readout
        if batch_idx is None:
            # Single molecule
            h_mean = h.mean(dim=0, keepdim=True)
            h_max  = h.max(dim=0).values.unsqueeze(0)
        else:
            n_graphs = batch_idx.max().item() + 1
            h_mean = torch.zeros(n_graphs, self.node_dim, device=h.device)
            h_max  = torch.full((n_graphs, self.node_dim), float('-inf'), device=h.device)
            h_mean.scatter_add_(0, batch_idx.unsqueeze(-1).expand_as(h), h)
            h_max  = torch.zeros_like(h_mean)
            for g in range(n_graphs):
                mask = batch_idx == g
                if mask.any():
                    h_mean[g] = h[mask].mean(0)
                    h_max[g]  = h[mask].max(0).values

        # Concatenate mean + max
        graph_feat = torch.cat([h_mean, h_max], dim=-1)
        return self.readout(graph_feat)

    def predict_smiles(self, smiles: str, device: str = 'cpu') -> Optional[float]:
        """Single-molecule inference from SMILES string."""
        g = smiles_to_graph(smiles)
        if g is None:
            return None
        self.eval()
        with torch.no_grad():
            nf = g['node_feats'].to(device)
            ei = g['edge_index'].to(device)
            ef = g['edge_feats'].to(device)
            logS = self(nf, ei, ef).item()
        return round(logS, 3)

    def save(self, path: str, config: dict = None):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': config or {
                'node_in': self.node_in, 'edge_in': self.edge_in,
                'node_dim': self.node_dim, 'n_layers': self.n_layers,
                'hidden_dim': self.hidden_dim,
            }
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'MPNNSolubility':
        ckpt = torch.load(path, map_location=device)
        model = cls(**ckpt['config'])
        model.load_state_dict(ckpt['model_state_dict'])
        return model.to(device).eval()


# ── DATASET ──────────────────────────────────────────────────────────────────
class SolubilityDataset(torch.utils.data.Dataset):
    """
    Dataset from your final_unique_train.csv / final_unique_test.csv.
    Expects columns: 'SMILES', 'logS' (or 'Solubility').
    """
    def __init__(self, csv_path: str, smiles_col: str = 'SMILES',
                 target_col: str = None):
        df = pd.read_csv(csv_path)

        # Auto-detect target column
        if target_col is None:
            for c in ['logS','LogS','Solubility','solubility','log_s','log S']:
                if c in df.columns:
                    target_col = c
                    break
            if target_col is None:
                raise ValueError(f"Could not find target column. Available: {list(df.columns)}")

        print(f"Dataset: {len(df)} rows | target='{target_col}'")

        self.graphs = []
        self.targets = []
        self.smiles_list = []
        skipped = 0

        for _, row in df.iterrows():
            smi = str(row[smiles_col])
            tgt = float(row[target_col])
            g = smiles_to_graph(smi)
            if g is None:
                skipped += 1
                continue
            self.graphs.append(g)
            self.targets.append(tgt)
            self.smiles_list.append(smi)

        print(f"Valid: {len(self.graphs)} | Skipped: {skipped}")

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], torch.tensor([self.targets[idx]], dtype=torch.float)


def collate_graphs(batch):
    """Custom collate for variable-size molecular graphs."""
    graphs, targets = zip(*batch)

    # Offset edge indices for each graph in the batch
    node_feats_list, edge_index_list, edge_feats_list, batch_idx_list = [], [], [], []
    node_offset = 0

    for i, g in enumerate(graphs):
        n = g['n_atoms']
        node_feats_list.append(g['node_feats'])
        if g['edge_index'].size(1) > 0:
            edge_index_list.append(g['edge_index'] + node_offset)
        edge_feats_list.append(g['edge_feats'])
        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))
        node_offset += n

    node_feats = torch.cat(node_feats_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros(2, 0, dtype=torch.long)
    edge_feats = torch.cat(edge_feats_list, dim=0) if any(e.size(0) > 0 for e in edge_feats_list) else torch.zeros(0, BOND_DIM)
    batch_idx  = torch.cat(batch_idx_list, dim=0)
    targets    = torch.cat(targets, dim=0)

    return node_feats, edge_index, edge_feats, batch_idx, targets


# ── TRAINING ─────────────────────────────────────────────────────────────────
def train_mpnn(
    train_csv: str,
    test_csv:  str,
    output_dir: str = 'models',
    node_dim:   int = 128,
    n_layers:   int = 4,
    hidden_dim: int = 256,
    dropout:    float = 0.1,
    lr:         float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    epochs:     int = 100,
    patience:   int = 15,
    device:     str = None,
):
    """
    Full training pipeline for MPNN solubility model.

    Usage:
        from mpnn_model import train_mpnn
        train_mpnn('final_unique_train.csv', 'final_unique_test.csv')
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # Datasets
    train_ds = SolubilityDataset(train_csv)
    test_ds  = SolubilityDataset(test_csv)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_graphs, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_graphs, num_workers=0
    )

    # Model
    config = dict(node_dim=node_dim, n_layers=n_layers,
                  hidden_dim=hidden_dim, dropout=dropout)
    model = MPNNSolubility(**config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    no_improve = 0
    history = {'train_rmse': [], 'val_rmse': [], 'val_mae': [], 'val_r2': []}

    for epoch in range(epochs):
        # ── TRAIN ──
        model.train()
        train_loss = 0.0
        for nf, ei, ef, bi, targets in train_loader:
            nf, ei, ef, bi, targets = (
                nf.to(device), ei.to(device), ef.to(device),
                bi.to(device), targets.to(device)
            )
            optimizer.zero_grad()
            pred = model(nf, ei, ef, bi).squeeze()
            loss = criterion(pred, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(targets)

        train_rmse = (train_loss / len(train_ds)) ** 0.5

        # ── VALIDATE ──
        model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for nf, ei, ef, bi, targets in test_loader:
                nf, ei, ef, bi = nf.to(device), ei.to(device), ef.to(device), bi.to(device)
                pred = model(nf, ei, ef, bi).squeeze().cpu()
                preds_all.append(pred)
                targets_all.append(targets)

        preds   = torch.cat(preds_all).numpy()
        targets = torch.cat(targets_all).numpy()

        val_mse  = np.mean((preds - targets) ** 2)
        val_rmse = val_mse ** 0.5
        val_mae  = np.mean(np.abs(preds - targets))
        ss_res   = np.sum((targets - preds) ** 2)
        ss_tot   = np.sum((targets - targets.mean()) ** 2)
        val_r2   = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        scheduler.step(val_rmse)

        history['train_rmse'].append(round(train_rmse, 4))
        history['val_rmse'].append(round(val_rmse, 4))
        history['val_mae'].append(round(val_mae, 4))
        history['val_r2'].append(round(val_r2, 4))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train RMSE: {train_rmse:.4f} | "
                  f"Val RMSE: {val_rmse:.4f} | "
                  f"Val MAE: {val_mae:.4f} | "
                  f"Val R²: {val_r2:.4f}")

        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            no_improve = 0
            model.save(
                os.path.join(output_dir, 'mpnn_solubility_best.pt'),
                config=config
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.save(os.path.join(output_dir, 'mpnn_solubility_final.pt'), config=config)

    # Save history
    with open(os.path.join(output_dir, 'mpnn_training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest Val RMSE: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}/mpnn_solubility_best.pt")
    return history


# ── EVALUATION ───────────────────────────────────────────────────────────────
def evaluate_mpnn(model_path: str, test_csv: str, device: str = 'cpu'):
    """
    Evaluate trained MPNN against your test set.
    Compares to XGBoost results.
    """
    model = MPNNSolubility.load(model_path, device)
    ds    = SolubilityDataset(test_csv)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=False, collate_fn=collate_graphs
    )

    preds, targets = [], []
    with torch.no_grad():
        for nf, ei, ef, bi, tgt in loader:
            nf, ei, ef, bi = nf.to(device), ei.to(device), ef.to(device), bi.to(device)
            p = model(nf, ei, ef, bi).squeeze().cpu().numpy()
            preds.extend(p.tolist())
            targets.extend(tgt.numpy().tolist())

    preds   = np.array(preds)
    targets = np.array(targets)

    rmse = np.mean((preds - targets) ** 2) ** 0.5
    mae  = np.mean(np.abs(preds - targets))
    ss   = np.sum((targets - preds) ** 2)
    ss_t = np.sum((targets - targets.mean()) ** 2)
    r2   = 1 - ss / ss_t

    print("\n" + "="*50)
    print("MPNN Solubility Model — Test Set Evaluation")
    print("="*50)
    print(f"  RMSE:  {rmse:.4f} log(mol/L)")
    print(f"  MAE:   {mae:.4f} log(mol/L)")
    print(f"  R²:    {r2:.4f}")
    print(f"  N:     {len(targets)}")
    print("\nComparison reference (XGBoost on same dataset):")
    print("  Target: RMSE < 1.0, R² > 0.80")
    print("="*50)

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'n': len(targets)}


# ── ENSEMBLE PREDICTOR ───────────────────────────────────────────────────────
class EnsemblePredictor:
    """
    Ensemble of XGBoost + MPNN for improved solubility prediction.
    Combines both models with configurable weights.

    Usage:
        ensemble = EnsemblePredictor(
            xgb_model_path='xgboost_model_298_4045.json',
            mpnn_model_path='models/mpnn_solubility_best.pt'
        )
        logS, contributions = ensemble.predict('CCO')
    """
    def __init__(
        self,
        xgb_model_path:  str,
        mpnn_model_path: str,
        xgb_weight:  float = 0.5,
        mpnn_weight: float = 0.5,
        device: str = 'cpu'
    ):
        self.xgb_weight  = xgb_weight
        self.mpnn_weight = mpnn_weight
        self.device = device

        # Load XGBoost
        try:
            import xgboost as xgb
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(xgb_model_path)
            self.xgb_ok = True
            print(f"✅ XGBoost loaded from {xgb_model_path}")
        except Exception as e:
            self.xgb_ok = False
            print(f"⚠ XGBoost not loaded: {e}")

        # Load MPNN
        try:
            self.mpnn_model = MPNNSolubility.load(mpnn_model_path, device)
            self.mpnn_ok = True
            print(f"✅ MPNN loaded from {mpnn_model_path}")
        except Exception as e:
            self.mpnn_ok = False
            print(f"⚠ MPNN not loaded: {e}")

    def predict_xgb(self, smiles: str) -> Optional[float]:
        """XGBoost prediction using 298-feature pipeline."""
        if not self.xgb_ok: return None
        try:
            from solubility_api import compute_all_features
            feats = compute_all_features(smiles)
            return float(self.xgb_model.predict(feats)[0])
        except Exception as e:
            print(f"XGBoost error: {e}")
            return None

    def predict_mpnn(self, smiles: str) -> Optional[float]:
        """MPNN prediction from molecular graph."""
        if not self.mpnn_ok: return None
        return self.mpnn_model.predict_smiles(smiles, self.device)

    def predict(self, smiles: str) -> dict:
        """
        Ensemble prediction combining both models.

        Returns dict with:
          logS_xgb, logS_mpnn, logS_ensemble,
          confidence, model_agreement
        """
        xgb_pred  = self.predict_xgb(smiles)
        mpnn_pred = self.predict_mpnn(smiles)

        # Compute ensemble
        if xgb_pred is not None and mpnn_pred is not None:
            ensemble = (self.xgb_weight * xgb_pred +
                        self.mpnn_weight * mpnn_pred)
            agreement = abs(xgb_pred - mpnn_pred)
            # High confidence when models agree
            confidence = max(0, min(1, 1.0 - agreement / 3.0))
        elif xgb_pred is not None:
            ensemble = xgb_pred; confidence = 0.7
        elif mpnn_pred is not None:
            ensemble = mpnn_pred; confidence = 0.7
        else:
            return {'error': 'Both models failed', 'smiles': smiles}

        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        mw  = Descriptors.MolWt(mol) if mol else None

        def classify(logS):
            if logS > 0:   return 'highly_soluble','Highly Soluble'
            if logS > -2:  return 'soluble','Soluble'
            if logS > -4:  return 'slightly_soluble','Slightly Soluble'
            return 'practically_insoluble','Practically Insoluble'

        cls, lbl = classify(ensemble)

        return {
            'smiles': smiles,
            'logS_xgboost':  round(xgb_pred, 3)  if xgb_pred  is not None else None,
            'logS_mpnn':     round(mpnn_pred, 3)  if mpnn_pred is not None else None,
            'logS_ensemble': round(ensemble, 3),
            'mol_per_liter': round(10**ensemble, 4),
            'gram_per_liter': round((10**ensemble)*mw, 4) if mw else None,
            'molecular_weight': round(mw, 2) if mw else None,
            'solubility_class': cls,
            'solubility_label': lbl,
            'model_agreement':  round(agreement, 3) if xgb_pred and mpnn_pred else None,
            'confidence': round(confidence, 3),
            'in_domain': (mw or 0) < 600,
            'domain_score': max(0, min(100, int(100*(1-max(0,(mw or 0)-100)/700)))) if mw else 50,
            'models_used': {
                'xgboost': self.xgb_ok,
                'mpnn': self.mpnn_ok,
                'weights': {'xgb': self.xgb_weight, 'mpnn': self.mpnn_weight}
            }
        }


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MPNN Solubility Model')
    sub = parser.add_subparsers(dest='command')

    # Train
    tr = sub.add_parser('train')
    tr.add_argument('--train_csv', default='final_unique_train.csv')
    tr.add_argument('--test_csv',  default='final_unique_test.csv')
    tr.add_argument('--output',    default='models')
    tr.add_argument('--epochs',    type=int, default=100)
    tr.add_argument('--batch',     type=int, default=32)
    tr.add_argument('--lr',        type=float, default=1e-3)

    # Evaluate
    ev = sub.add_parser('evaluate')
    ev.add_argument('--model', required=True)
    ev.add_argument('--test_csv', default='final_unique_test.csv')

    # Predict
    pr = sub.add_parser('predict')
    pr.add_argument('--model', required=True)
    pr.add_argument('--smiles', required=True)

    args = parser.parse_args()

    if args.command == 'train':
        train_mpnn(args.train_csv, args.test_csv, args.output,
                   epochs=args.epochs, batch_size=args.batch, lr=args.lr)

    elif args.command == 'evaluate':
        evaluate_mpnn(args.model, args.test_csv)

    elif args.command == 'predict':
        model = MPNNSolubility.load(args.model)
        logS = model.predict_smiles(args.smiles)
        print(f"SMILES: {args.smiles}")
        print(f"logS:   {logS}")
        print(f"mol/L:  {10**logS:.4f}")
