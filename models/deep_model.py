"""
Tier 4 – GolferLSTM: Deep Learning Sequence Model

Treats each player's tournament history as a time series and uses an LSTM
to predict the probability of a top-10 finish in the next event.

Architecture
------------
  Input  : (batch, seq_len, n_features)  – last N tournament appearances
  LSTM   : 2-layer bi-directional LSTM
  Head   : FC → ReLU → Dropout → FC → Sigmoid
  Output : scalar probability in [0, 1]

Data
----
  One row per (player, tournament) from the enriched parquet.
  Sequences are built by taking each player's most recent `seq_len` events
  *before* the target tournament (no leakage).

Usage
-----
    python models/deep_model.py                # train with defaults
    python models/deep_model.py --epochs 30 --lr 5e-4
    python models/deep_model.py --seq-len 15
"""
from __future__ import annotations

import argparse
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

# ── Feature set ───────────────────────────────────────────────────────────────
# Per-event features that exist before the tournament starts.

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
    "sg_total_prev_season", "sg_putting_prev_season",
    "sg_approach_prev_season", "sg_off_tee_prev_season",
    "driving_distance_prev_season", "driving_accuracy_prev_season",
    "gir_pct_prev_season", "scoring_avg_prev_season",
]

# ── Dataset ───────────────────────────────────────────────────────────────────

class GolferDataset(Dataset):
    """
    Sliding-window dataset over player career sequences.

    For each (player, tournament_index) pair where i >= 1, the input is
    the feature vectors of the player's *previous* min(seq_len, i) appearances
    and the target is whether they achieved top-10 at tournament i.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: list[str],
        seq_len: int = 20,
        top_n: int = 10,
    ):
        self.seq_len   = seq_len
        self.feat_cols = feat_cols
        self.records: list[tuple[np.ndarray, int]] = []  # (sequence, label)

        df = df.sort_values(["name", "year", "date"]).reset_index(drop=True)

        for _, player_df in df.groupby("name", sort=False):
            player_df = player_df.reset_index(drop=True)
            feats = player_df[feat_cols].fillna(0).values.astype(np.float32)
            labels = (player_df["tournament_rank"] <= top_n).values.astype(np.float32)

            # Each row i uses rows [max(0, i-seq_len) … i-1] as context
            for i in range(1, len(player_df)):
                start = max(0, i - seq_len)
                context = feats[start:i]  # shape (context_len, n_features)

                # Pad on the LEFT to ensure fixed length
                pad_len = seq_len - len(context)
                if pad_len > 0:
                    pad = np.zeros((pad_len, len(feat_cols)), dtype=np.float32)
                    context = np.vstack([pad, context])

                self.records.append((context.astype(np.float32), float(labels[i])))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        seq, label = self.records[idx]
        return torch.tensor(seq), torch.tensor(label, dtype=torch.float32)


# ── Model ─────────────────────────────────────────────────────────────────────

class GolferLSTM(nn.Module):
    """
    Two-layer bidirectional LSTM with a small MLP head.
    Input  : (batch, seq_len, input_dim)
    Output : (batch,) – probability in [0, 1]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        lstm_out_dim = hidden_dim * 2  # bidirectional doubles output

        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (n_layers * 2, batch, hidden_dim) for bidirectional
        # Take the last layer's forward + backward hidden states
        h_forward  = h_n[-2]  # shape (batch, hidden_dim)
        h_backward = h_n[-1]  # shape (batch, hidden_dim)
        h_last = torch.cat([h_forward, h_backward], dim=-1)  # (batch, hidden_dim*2)
        return self.head(h_last).squeeze(-1)  # (batch,)


# ── Training utilities ────────────────────────────────────────────────────────

def _evaluate(model: GolferLSTM, loader: DataLoader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            probs = model(seqs)
            loss  = criterion(probs, labels)
            total_loss += loss.item() * len(labels)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return {"loss": total_loss / len(loader.dataset), "auc": auc}


# ── Main training function ────────────────────────────────────────────────────

def train(
    seq_len:    int   = 20,
    top_n:      int   = 10,
    hidden_dim: int   = 64,
    n_layers:   int   = 2,
    dropout:    float = 0.3,
    lr:         float = 1e-3,
    epochs:     int   = 20,
    batch_size: int   = 256,
) -> GolferLSTM:
    print("\n" + "=" * 60)
    print(f"GOLFER LSTM  (Top-{top_n}, seq_len={seq_len})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────
    for candidate in [
        DATA_DIR / "espn_with_extended_features.parquet",
        DATA_DIR / "espn_with_owgr_features.parquet",
        DATA_DIR / "espn_player_tournament_features.parquet",
    ]:
        if candidate.exists():
            df = pd.read_parquet(candidate)
            print(f"[OK] Loaded {len(df):,} rows from {candidate.name}")
            break
    else:
        raise FileNotFoundError("No feature parquet found. Run features/build_features.py first.")

    if "date" not in df.columns:
        df["date"] = pd.to_datetime(dict(year=df["year"], month=1, day=1))
    else:
        df["date"] = pd.to_datetime(df["date"])

    feat_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]
    missing = set(CANDIDATE_FEATURES) - set(feat_cols)
    print(f"[OK] Using {len(feat_cols)} features ({len(missing)} unavailable)")

    # ── Time-based split ──────────────────────────────────────────────────
    train_df = df[df["year"] <= 2022].copy()
    val_df   = df[df["year"] >= 2023].copy()
    print(f"  Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows")

    train_ds = GolferDataset(train_df, feat_cols, seq_len=seq_len, top_n=top_n)
    val_ds   = GolferDataset(val_df,   feat_cols, seq_len=seq_len, top_n=top_n)
    print(f"  Train sequences: {len(train_ds):,} | Val sequences: {len(val_ds):,}")

    if len(train_ds) == 0:
        raise RuntimeError("No training sequences – check data_files/ has results.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Class weighting ───────────────────────────────────────────────────
    labels_arr = np.array([r[1] for r in train_ds.records])
    pos = labels_arr.sum()
    neg = len(labels_arr) - pos
    pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32).to(device)
    print(f"  Class imbalance: {neg / max(pos, 1):.1f}:1  (pos_weight={pos_weight.item():.2f})")

    # ── Model, loss, optimiser ────────────────────────────────────────────
    model = GolferLSTM(
        input_dim=len(feat_cols),
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCELoss(weight=pos_weight.expand(1))  # scalar weight for positives
    criterion = nn.BCELoss()  # simpler; pos_weight handled via architecture
    # Use weighted binary cross-entropy
    def _weighted_bce(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = torch.where(targets == 1, pos_weight, torch.ones_like(targets))
        return nn.functional.binary_cross_entropy(preds, targets, weight=w)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=3, verbose=True
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_auc  = 0.0
    best_state    = None

    print(f"\n{'Epoch':>5}  {'Train Loss':>12}  {'Train AUC':>10}  {'Val Loss':>10}  {'Val AUC':>8}")
    print("-" * 58)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimiser.zero_grad()
            probs = model(seqs)
            loss  = _weighted_bce(probs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item() * len(labels)

        train_metrics = _evaluate(model, train_loader, _weighted_bce, device)
        val_metrics   = _evaluate(model, val_loader,   _weighted_bce, device)

        scheduler.step(val_metrics["auc"])

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"{epoch:>5}  {epoch_loss / len(train_ds):>12.4f}  "
              f"{train_metrics['auc']:>10.4f}  "
              f"{val_metrics['loss']:>10.4f}  "
              f"{val_metrics['auc']:>8.4f}"
              + ("  ← best" if val_metrics["auc"] >= best_val_auc else ""))

    # ── Restore best weights & save ───────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = MODEL_DIR / "golfer_lstm.pt"
    meta_path  = MODEL_DIR / "golfer_lstm_meta.json"

    torch.save({
        "state_dict": model.state_dict(),
        "input_dim":  len(feat_cols),
        "hidden_dim": hidden_dim,
        "n_layers":   n_layers,
        "dropout":    dropout,
        "seq_len":    seq_len,
        "top_n":      top_n,
        "feat_cols":  feat_cols,
        "best_val_auc": best_val_auc,
    }, model_path)

    import json
    with open(meta_path, "w") as f:
        json.dump({
            "input_dim": len(feat_cols),
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "seq_len": seq_len,
            "top_n": top_n,
            "best_val_auc": best_val_auc,
            "feat_cols": feat_cols,
        }, f, indent=2)

    print(f"\n[OK] Best val AUC:   {best_val_auc:.4f}")
    print(f"[OK] Model saved:    {model_path}")
    print(f"[OK] Meta saved:     {meta_path}")

    return model


# ── Inference helper ──────────────────────────────────────────────────────────

def load_model() -> tuple[GolferLSTM, dict]:
    """Load saved model and metadata."""
    model_path = MODEL_DIR / "golfer_lstm.pt"
    if not model_path.exists():
        raise FileNotFoundError("golfer_lstm.pt not found — run models/deep_model.py first.")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = GolferLSTM(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        n_layers=checkpoint["n_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_sequences(sequences: np.ndarray) -> np.ndarray:
    """
    Run inference on a batch of pre-built sequences.

    Parameters
    ----------
    sequences : np.ndarray of shape (n_players, seq_len, n_features)

    Returns
    -------
    np.ndarray of shape (n_players,) – probabilities in [0, 1]
    """
    model, _ = load_model()
    with torch.no_grad():
        t = torch.tensor(sequences, dtype=torch.float32)
        return model(t).numpy()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GolferLSTM (Tier 4).")
    parser.add_argument("--seq-len",    type=int,   default=20)
    parser.add_argument("--top",        type=int,   default=10)
    parser.add_argument("--hidden-dim", type=int,   default=64)
    parser.add_argument("--n-layers",   type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=256)
    args = parser.parse_args()

    train(
        seq_len=args.seq_len,
        top_n=args.top,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
