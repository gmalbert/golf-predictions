"""
Tier 5 – GolferTransformer: Transformer Encoder for player sequences

Treats each player's recent N tournaments as a sequence of feature vectors
and applies a standard Transformer encoder followed by a winner-probability head.

Architecture
------------
  Input  : (batch, seq_len, n_features)  – last N tournament appearances
  Linear : project to d_model
  PositionalEmbedding : learnable position embeddings
  TransformerEncoder  : n_heads × n_layers
  Pool   : mean of output tokens
  Head   : FC → LayerNorm → GELU → Dropout → FC → Sigmoid
  Output : scalar win probability in [0, 1]

Usage
-----
    python models/transformer_golfer.py                 # train default
    python models/transformer_golfer.py --epochs 30
    python models/transformer_golfer.py --compare-lstm  # compare vs. LSTM
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_files"
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

CANDIDATE_FEATURES = [
    "prior_avg_score", "prior_avg_score_5", "prior_avg_score_10",
    "prior_std_score", "prior_std_score_5", "prior_std_score_10",
    "prior_top10_rate_5", "prior_top10_rate_10",
    "prior_count",
    "last_event_score", "last_event_rank",
    "days_since_last_event",
    "career_best_rank",
    "tournaments_last_365d", "season_to_date_avg_score",
    "played_last_30d",
    "course_history_avg_score",
    "owgr_rank_current", "owgr_rank_4w_ago", "owgr_rank_12w_ago",
    "owgr_points_current",
    "owgr_rank_change_4w", "owgr_rank_change_12w",
    "is_major", "is_playoff", "purse_tier",
    "course_type_enc", "grass_type_enc",
    "field_strength", "field_size",
]


# ── Dataset ──────────────────────────────────────────────────────────────────

class GolferSequenceDataset(Dataset):
    """Build per-player career sequences for the transformer."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_len: int = 10,
        target_col: str = "won",
    ):
        self.sequences: list[np.ndarray] = []
        self.labels: list[float] = []
        self.seq_len = seq_len

        df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
        X = df[feature_cols].fillna(0).values.astype(np.float32)

        # Z-score normalise column-wise
        mean = X.mean(0, keepdims=True)
        std = X.std(0, keepdims=True) + 1e-7
        X = (X - mean) / std
        self._mean, self._std = mean, std

        for pid, grp in df.groupby("player_id"):
            idx = grp.index.tolist()
            for i in range(len(idx)):
                end = i  # predict event at idx[i]
                start = max(0, end - seq_len)
                hist_idx = idx[start:end]
                label = float(df.at[idx[i], target_col])

                if len(hist_idx) == 0:
                    seq = np.zeros((seq_len, len(feature_cols)), dtype=np.float32)
                else:
                    hist = X[hist_idx]
                    pad = seq_len - len(hist)
                    if pad > 0:
                        hist = np.vstack([np.zeros((pad, len(feature_cols)), dtype=np.float32), hist])
                    seq = hist[-seq_len:]

                self.sequences.append(seq)
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ── Model ────────────────────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.emb(pos)


class GolferTransformer(nn.Module):
    """
    Transformer encoder that maps a player's recent tournament sequence
    to a win-probability estimate for their next event.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.15,
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = LearnablePositionalEncoding(d_model, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Mean-pool over sequence dimension
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


# ── Training loop ───────────────────────────────────────────────────────────

def _load_data():
    for candidate in [
        DATA_DIR / "espn_with_extended_features.parquet",
        DATA_DIR / "espn_with_owgr_features.parquet",
        DATA_DIR / "espn_player_tournament_features.parquet",
    ]:
        if candidate.exists():
            df = pd.read_parquet(candidate)
            print(f"[OK] Loaded {len(df):,} rows from {candidate.name}")
            return df
    raise FileNotFoundError("No feature parquet found in data_files/")


def train(epochs: int = 20, seq_len: int = 10, batch_size: int = 512,
          lr: float = 1e-3, compare_lstm: bool = False):
    df = _load_data()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.dropna(subset=["player_id", "tournament_rank"])
    df["won"] = (df["tournament_rank"] == 1).astype(int)

    feature_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]
    print(f"[INFO] Using {len(feature_cols)} features, seq_len={seq_len}")

    # Time-based split
    df = df.sort_values("date")
    n = len(df)
    val_cut = int(n * 0.85)
    train_df, val_df = df.iloc[:val_cut], df.iloc[val_cut:]

    train_ds = GolferSequenceDataset(train_df, feature_cols, seq_len)
    val_ds   = GolferSequenceDataset(val_df,   feature_cols, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}")

    model = GolferTransformer(input_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = torch.tensor(
        [(len(train_ds) - sum(s[1].item() for s in train_ds)) /
         max(1, sum(s[1].item() for s in train_ds))],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Replace Sigmoid in head with identity for BCEWithLogitsLoss compatibility
    model.head[-1] = nn.Identity()

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(yb)

        avg_train = total_loss / len(train_ds)
        avg_val   = val_loss / len(val_ds)
        print(f"Epoch {epoch:3d}/{epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = MODEL_DIR / "transformer_golfer.pt"
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "seq_len": seq_len,
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
            }, save_path)
            print(f"  ✓ Saved best model → {save_path}")

    if compare_lstm:
        print("\n[INFO] Run `python models/deep_model.py` to train the LSTM for comparison.")

    print(f"\n[Done] Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GolferTransformer")
    parser.add_argument("--epochs",   type=int, default=20)
    parser.add_argument("--seq-len",  type=int, default=10)
    parser.add_argument("--batch",    type=int, default=512)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--compare-lstm", action="store_true")
    args = parser.parse_args()
    train(args.epochs, args.seq_len, args.batch, args.lr, args.compare_lstm)
