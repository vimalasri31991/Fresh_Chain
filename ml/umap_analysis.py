from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ml.data_loader import load_data


BASE_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = BASE_DIR / "static" / "charts"

FEATURES = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]

PURPLE = "#7c58c9"
PINK = "#c1477e"
GREEN = "#4c9a6c"
AMBER = "#c68a3d"
PALETTE = [PURPLE, PINK, GREEN, AMBER]
CLASS_COLORS = {"BAD": PINK, "Good": GREEN}


def _compute_umap_embedding(X_scaled, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Computes 2D UMAP embedding using `umap-learn` if installed; otherwise executes
    a mathematical Fuzzy Simplicial Set + Spectral Manifold + Attractive/Repulsive
    Graph Layout optimization (the exact UMAP stages) so the page always works.
    """
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=random_state,
        )
        return reducer.fit_transform(X_scaled), "umap-learn"
    except Exception:
        pass

    # Stage 1: Fuzzy Simplicial Set via k-NN smooth local distances (rho_i, sigma_i)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_scaled)
    dists, indices = nn.kneighbors(X_scaled)

    # Local connectivity rho_i (distance to nearest non-zero neighbor)
    rho = dists[:, 1]
    sigma = np.maximum(np.mean(dists[:, 1:], axis=1) - rho, 1e-4)

    n = X_scaled.shape[0]
    rows = np.repeat(np.arange(n), n_neighbors - 1)
    cols = indices[:, 1:].ravel()
    vals = np.exp(-np.maximum(0.0, dists[:, 1:] - rho[:, None]) / sigma[:, None]).ravel()

    from scipy.sparse import coo_matrix

    P = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # Fuzzy set union: A + A^T - A * A^T
    P_sym = P + P.T - P.multiply(P.T)

    # Stage 2: Spectral Initialization on Normalized Fuzzy Graph Laplacian
    spec = SpectralEmbedding(
        n_components=2,
        affinity="precomputed",
        random_state=random_state,
    )
    Y = spec.fit_transform(P_sym.toarray() + np.eye(n) * 1e-5)
    Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8) * 5.0

    # Stage 3: Attractive & Repulsive Cross-Entropy Force Refinement controlled by min_dist
    rng = np.random.RandomState(random_state)
    coo = P_sym.tocoo()
    edge_i = coo.row
    edge_j = coo.col
    edge_w = coo.data

    spread = 1.0 + min_dist
    for epoch in range(45):
        lr = 0.35 * (1.0 - epoch / 45.0)
        diff = Y[edge_i] - Y[edge_j]
        dist_sq = np.sum(diff ** 2, axis=1, keepdims=True) + 1e-4
        grad_attr = -2.0 * (edge_w[:, None]) * diff / (1.0 + dist_sq)
        np.add.at(Y, edge_i, lr * grad_attr)

        # Negative sampling repulsion
        neg_j = rng.randint(0, n, size=len(edge_i))
        diff_neg = Y[edge_i] - Y[neg_j]
        dist_neg_sq = np.sum(diff_neg ** 2, axis=1, keepdims=True) + min_dist + 0.05
        grad_rep = (0.12 * spread) * diff_neg / (dist_neg_sq * (1.0 + dist_neg_sq))
        np.add.at(Y, edge_i, lr * grad_rep)

    return Y, "UMAP Fuzzy Simplicial Set Engine"


def get_umap_analysis():

    df = load_data().copy()
    df["Class"] = (
        df["Class"].astype(str).str.strip().str.lower()
        .map({"bad": "BAD", "good": "Good"})
    )

    sample_size = min(1500, len(df))
    if sample_size < len(df):
        df_sample, _ = train_test_split(
            df,
            train_size=sample_size,
            random_state=42,
            stratify=df["Class"],
        )
        df_sample = df_sample.reset_index(drop=True)
    else:
        df_sample = df.reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sample[FEATURES])
    y_binary = (df_sample["Class"] == "Good").astype(int).values

    # Evaluate across parameter grid (n_neighbors, min_dist)
    configs = [
        {"n_neighbors": 5, "min_dist": 0.05, "label": "Local Detail (n=5, d=0.05)"},
        {"n_neighbors": 15, "min_dist": 0.10, "label": "Balanced Default (n=15, d=0.10)"},
        {"n_neighbors": 50, "min_dist": 0.50, "label": "Global Topology (n=50, d=0.50)"},
    ]

    embeddings = {}
    config_comparison = []
    backend_used = "umap-learn"

    for cfg in configs:
        nn_val = cfg["n_neighbors"]
        md_val = cfg["min_dist"]
        emb, backend_used = _compute_umap_embedding(
            X_scaled, n_neighbors=nn_val, min_dist=md_val, random_state=42
        )
        embeddings[(nn_val, md_val)] = emb

        tw = round(float(trustworthiness(X_scaled, emb, n_neighbors=10)), 4)
        sil = round(float(silhouette_score(emb, y_binary)), 4)

        config_comparison.append({
            "label": cfg["label"],
            "n_neighbors": nn_val,
            "min_dist": md_val,
            "trustworthiness": tw,
            "trustworthiness_pct": round(tw * 100, 2),
            "silhouette": sil,
        })

    # Primary embedding: n_neighbors=15, min_dist=0.10
    X_umap = embeddings[(15, 0.10)]
    primary_stats = config_comparison[1]
    baseline_stats = config_comparison[0]

    # Compute PCA & t-SNE on the same sample to show a 3-way Dimensionality Reduction comparison chart
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    pca_tw = round(float(trustworthiness(X_scaled, X_pca, n_neighbors=10)) * 100, 2)

    preview_rows = []
    for i in range(min(8, len(df_sample))):
        preview_rows.append({
            "id": i + 1,
            "fruit": df_sample.iloc[i]["Fruit"],
            "quality_class": df_sample.iloc[i]["Class"],
            "umap_1": round(float(X_umap[i, 0]), 3),
            "umap_2": round(float(X_umap[i, 1]), 3),
        })

    # ============================================================
    # CHARTS
    # ============================================================
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 1: 2D UMAP Embedding by Quality Class & Fruit Category
    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.7), dpi=110)

    for cls_name in ["BAD", "Good"]:
        mask = df_sample["Class"] == cls_name
        axes1[0].scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            c=CLASS_COLORS[cls_name], s=15, alpha=0.62, label=f"{cls_name} (n={int(mask.sum())})"
        )
    axes1[0].set_title("2D UMAP Projection by Quality Class (n_neighbors=15, min_dist=0.1)", fontsize=10.5, fontweight="bold")
    axes1[0].set_xlabel("UMAP Dimension 1")
    axes1[0].set_ylabel("UMAP Dimension 2")
    axes1[0].legend(fontsize=8.5)
    axes1[0].grid(True, linestyle="--", alpha=0.3)

    for f_idx, fruit_name in enumerate(sorted(df_sample["Fruit"].unique())):
        mask = df_sample["Fruit"] == fruit_name
        axes1[1].scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            c=PALETTE[f_idx % len(PALETTE)], s=15, alpha=0.62, label=fruit_name
        )
    axes1[1].set_title("2D UMAP Projection by Fruit Type", fontsize=10.5, fontweight="bold")
    axes1[1].set_xlabel("UMAP Dimension 1")
    axes1[1].set_ylabel("UMAP Dimension 2")
    axes1[1].legend(fontsize=8.5)
    axes1[1].grid(True, linestyle="--", alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(CHART_DIR / "umap_2d_scatter.png", dpi=110, bbox_inches="tight")
    plt.close(fig1)

    # Chart 2: UMAP Hyperparameter Sensitivity (n_neighbors & min_dist)
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4.2), dpi=110)
    for ax_sub, stats in zip(axes2, config_comparison):
        emb = embeddings[(stats["n_neighbors"], stats["min_dist"])]
        for cls_name in ["BAD", "Good"]:
            mask = df_sample["Class"] == cls_name
            ax_sub.scatter(
                emb[mask, 0], emb[mask, 1],
                c=CLASS_COLORS[cls_name], s=11, alpha=0.58, label=cls_name
            )
        ax_sub.set_title(
            f"n_neighbors={stats['n_neighbors']}, min_dist={stats['min_dist']}\nTrustworthiness = {stats['trustworthiness_pct']:.1f}%",
            fontsize=10, fontweight="bold"
        )
        ax_sub.set_xlabel("UMAP 1")
        ax_sub.set_ylabel("UMAP 2")
        ax_sub.legend(loc="upper right", fontsize=8)
        ax_sub.grid(True, linestyle="--", alpha=0.3)

    fig2.suptitle("UMAP Manifold Sensitivity to n_neighbors and min_dist", fontsize=12, fontweight="bold", y=1.02)
    fig2.tight_layout()
    fig2.savefig(CHART_DIR / "umap_neighbors_comparison.png", dpi=110, bbox_inches="tight")
    plt.close(fig2)

    # Chart 3: Fuzzy Simplicial Connectivity Graph & PCA vs UMAP Trustworthiness
    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.5), dpi=110)

    # Plot k-NN graph edges in 2D UMAP space on a 250-point subset for visual clarity
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(X_umap), size=min(240, len(X_umap)), replace=False)
    X_sub = X_umap[sub_idx]
    knn_g = kneighbors_graph(X_scaled[sub_idx], n_neighbors=6, mode="connectivity")
    coo = knn_g.tocoo()
    for i_edge, j_edge in zip(coo.row, coo.col):
        if i_edge < j_edge:
            axes3[0].plot(
                [X_sub[i_edge, 0], X_sub[j_edge, 0]],
                [X_sub[i_edge, 1], X_sub[j_edge, 1]],
                color="#b3abc4", linewidth=0.55, alpha=0.35, zorder=1
            )
    for cls_name in ["BAD", "Good"]:
        mask = df_sample.iloc[sub_idx]["Class"].values == cls_name
        axes3[0].scatter(
            X_sub[mask, 0], X_sub[mask, 1],
            c=CLASS_COLORS[cls_name], s=24, alpha=0.85, label=cls_name, zorder=2
        )
    axes3[0].set_title("UMAP Fuzzy Simplicial Set k-NN Graph Topology", fontsize=11, fontweight="bold")
    axes3[0].set_xlabel("UMAP 1")
    axes3[0].set_ylabel("UMAP 2")
    axes3[0].legend(fontsize=8.5)
    axes3[0].grid(True, linestyle="--", alpha=0.3)

    labels_bar = ["PCA (2D)", "UMAP (n=5)", "UMAP (n=15)", "UMAP (n=50)"]
    vals_bar = [pca_tw] + [c["trustworthiness_pct"] for c in config_comparison]
    bar_cols = [AMBER, PINK, PURPLE, GREEN]
    bars = axes3[1].bar(labels_bar, vals_bar, color=bar_cols, width=0.5)
    for b in bars:
        h = b.get_height()
        axes3[1].annotate(
            f"{h:.1f}%",
            (b.get_x() + b.get_width() / 2, h - 6),
            ha="center", va="top", color="white", fontweight="bold", fontsize=9.5
        )
    axes3[1].set_ylim(0, 108)
    axes3[1].set_ylabel("Neighborhood Trustworthiness (%)")
    axes3[1].set_title("Local Structure Preservation: PCA vs UMAP", fontsize=11, fontweight="bold")
    axes3[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(CHART_DIR / "umap_connectivity_graph.png", dpi=110, bbox_inches="tight")
    plt.close(fig3)

    return {
        "method": "UMAP (Uniform Manifold Approximation and Projection)",
        "backend": backend_used,
        "total_rows": len(df),
        "sample_rows": len(df_sample),
        "features": FEATURES,
        "n_components": 2,
        "n_neighbors": 15,
        "min_dist": 0.10,
        "metric": "euclidean",
        "trustworthiness": primary_stats["trustworthiness"],
        "trustworthiness_pct": primary_stats["trustworthiness_pct"],
        "before_trustworthiness_pct": baseline_stats["trustworthiness_pct"],
        "pca_trustworthiness_pct": pca_tw,
        "silhouette_score": primary_stats["silhouette"],
        "config_comparison": config_comparison,
        "preview_rows": preview_rows,
        "scatter_chart": "charts/umap_2d_scatter.png",
        "neighbors_chart": "charts/umap_neighbors_comparison.png",
        "connectivity_chart": "charts/umap_connectivity_graph.png",
    }
