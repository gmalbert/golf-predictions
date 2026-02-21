"""
Course Embedding Model – learn dense course representations from player performance.

Courses where similar players perform well receive similar embeddings.
Model: Matrix-factorisation (player × course → expected normalised finish rank),
inspired by collaborative filtering in recommendation systems.

Usage
-----
    python models/course_embeddings.py               # train & save
    python models/course_embeddings.py --cluster     # cluster courses after training
    python models/course_embeddings.py --embed-dim 32 --epochs 40
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


# ── Dataset ──────────────────────────────────────────────────────────────────

class PlayerCourseDataset(Dataset):
    """
    Each sample: (player_id_enc, course_id_enc, normalised_finish_rank).
    Normalised finish = finish_rank / field_size  ∈ (0, 1].
    Lower is better (winner = 1/field_size).
    """

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df = df.dropna(subset=["player_id", "tournament", "tournament_rank"])

        # Encode player and course IDs as integers
        self.player_enc = {p: i for i, p in enumerate(df["player_id"].unique())}
        self.course_enc = {c: i for i, c in enumerate(df["tournament"].unique())}

        df["player_enc"] = df["player_id"].map(self.player_enc)
        df["course_enc"] = df["tournament"].map(self.course_enc)

        # Normalise finish rank by field size (approximate via yearly tournament size)
        if "field_size" in df.columns:
            df["norm_finish"] = (df["tournament_rank"] / df["field_size"]).clip(0, 1)
        else:
            fs = df.groupby(["tournament", "year"])["tournament_rank"].transform("count")
            df["norm_finish"] = (df["tournament_rank"] / fs).clip(0, 1)

        df = df.dropna(subset=["norm_finish"])

        self.players = torch.tensor(df["player_enc"].values, dtype=torch.long)
        self.courses = torch.tensor(df["course_enc"].values, dtype=torch.long)
        self.targets = torch.tensor(df["norm_finish"].values, dtype=torch.float32)

        self.n_players = len(self.player_enc)
        self.n_courses = len(self.course_enc)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        return self.players[idx], self.courses[idx], self.targets[idx]


# ── Model ────────────────────────────────────────────────────────────────────

class CourseEmbeddingModel(nn.Module):
    """
    Bilinear matrix factorisation with bias terms.
    Predicts normalised finish rank for a (player, course) pair.
    """

    def __init__(self, n_players: int, n_courses: int, embed_dim: int = 16):
        super().__init__()
        self.player_embed = nn.Embedding(n_players, embed_dim)
        self.course_embed = nn.Embedding(n_courses, embed_dim)
        self.player_bias  = nn.Embedding(n_players, 1)
        self.course_bias  = nn.Embedding(n_courses, 1)
        self.global_bias  = nn.Parameter(torch.zeros(1))

        # Initialise with small values
        nn.init.normal_(self.player_embed.weight, std=0.01)
        nn.init.normal_(self.course_embed.weight, std=0.01)

    def forward(self, player_ids: torch.Tensor, course_ids: torch.Tensor) -> torch.Tensor:
        p = self.player_embed(player_ids)   # (B, embed_dim)
        c = self.course_embed(course_ids)   # (B, embed_dim)
        dot = (p * c).sum(dim=1, keepdim=True)  # (B, 1)
        pred = (
            dot
            + self.player_bias(player_ids)
            + self.course_bias(course_ids)
            + self.global_bias
        ).squeeze(1)
        return torch.sigmoid(pred)  # → (0, 1)

    def get_course_embeddings(self) -> np.ndarray:
        return self.course_embed.weight.detach().cpu().numpy()

    def get_player_embeddings(self) -> np.ndarray:
        return self.player_embed.weight.detach().cpu().numpy()


# ── Training ─────────────────────────────────────────────────────────────────

def _load_data() -> pd.DataFrame:
    for p in [
        DATA_DIR / "espn_with_extended_features.parquet",
        DATA_DIR / "espn_with_owgr_features.parquet",
        DATA_DIR / "espn_player_tournament_features.parquet",
    ]:
        if p.exists():
            df = pd.read_parquet(p)
            print(f"[OK] Loaded {len(df):,} rows from {p.name}")
            return df
    raise FileNotFoundError("No feature parquet found.")


def train(embed_dim: int = 16, epochs: int = 20, batch_size: int = 2048, lr: float = 1e-3):
    df = _load_data()
    ds = PlayerCourseDataset(df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CourseEmbeddingModel(ds.n_players, ds.n_courses, embed_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"[INFO] {ds.n_players} players, {ds.n_courses} courses, embed_dim={embed_dim}")

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for p_ids, c_ids, y in loader:
            p_ids, c_ids, y = p_ids.to(device), c_ids.to(device), y.to(device)
            opt.zero_grad()
            pred = model(p_ids, c_ids)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        print(f"Epoch {epoch:3d}/{epochs}  MSE={total / len(ds):.5f}")

    # Save embeddings & model
    save_path = MODEL_DIR / "course_embeddings.pt"
    torch.save({
        "model_state": model.state_dict(),
        "player_enc": ds.player_enc,
        "course_enc": ds.course_enc,
        "embed_dim": embed_dim,
        "n_players": ds.n_players,
        "n_courses": ds.n_courses,
    }, save_path)
    print(f"[OK] Saved → {save_path}")

    # Also export course embeddings as parquet for downstream use
    course_embs = model.get_course_embeddings()
    idx_to_course = {v: k for k, v in ds.course_enc.items()}
    emb_df = pd.DataFrame(
        course_embs,
        columns=[f"course_emb_{i}" for i in range(embed_dim)],
    )
    emb_df.insert(0, "tournament", [idx_to_course[i] for i in range(ds.n_courses)])
    out = DATA_DIR / "course_embeddings.parquet"
    emb_df.to_parquet(out, index=False)
    print(f"[OK] Course embeddings → {out} ({ds.n_courses} courses × {embed_dim} dims)")
    return model, ds


def cluster_courses(model: CourseEmbeddingModel, ds: PlayerCourseDataset, n_clusters: int = 8):
    """K-Means cluster courses by their learned embeddings and print groupings."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[WARN] scikit-learn not available for clustering.")
        return

    embs = model.get_course_embeddings()
    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embs_scaled)

    idx_to_course = {v: k for k, v in ds.course_enc.items()}
    clusters: dict[int, list[str]] = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(idx_to_course[i])

    print(f"\n── Course clusters (k={n_clusters}) ──")
    for lbl, names in sorted(clusters.items()):
        print(f"\n  Cluster {lbl + 1} ({len(names)} courses):")
        for name in sorted(names)[:15]:
            print(f"    {name}")
        if len(names) > 15:
            print(f"    … and {len(names) - 15} more")

    # Save cluster assignments
    cluster_df = pd.DataFrame({
        "tournament": [idx_to_course[i] for i in range(ds.n_courses)],
        "course_cluster": labels,
    })
    out = DATA_DIR / "course_clusters.parquet"
    cluster_df.to_parquet(out, index=False)
    print(f"\n[OK] Cluster assignments → {out}")
    return cluster_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train course embedding model")
    parser.add_argument("--embed-dim",  type=int, default=16)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--cluster",    action="store_true", help="Cluster courses after training")
    parser.add_argument("--n-clusters", type=int, default=8)
    args = parser.parse_args()

    model, ds = train(args.embed_dim, args.epochs, args.batch_size, args.lr)
    if args.cluster:
        cluster_courses(model, ds, args.n_clusters)
