from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)
from sklearn.neighbors import NearestNeighbors
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
PALETTE = [PURPLE, GREEN, AMBER, "#3b82f6", "#10b981"]


def prepare_data():

    df = load_data()

    X = df[FEATURES].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return df, X_scaled


def _generate_dbscan_charts(X, labels, eps, min_samples):

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=110)

    # Left: k-Distance Plot for Selecting Epsilon
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    axes[0].plot(k_distances, color=PURPLE, linewidth=1.8, label=f"{min_samples}-NN Distance Curve")
    axes[0].axhline(eps, color=PINK, linestyle="--", linewidth=1.6, label=f"Chosen eps = {eps}")
    axes[0].set_title(f"k-Distance Graph (min_samples = {min_samples})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Points Sorted by Distance")
    axes[0].set_ylabel(f"{min_samples}-th Nearest Neighbor Distance")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Right: 2D PCA Projection of DBSCAN Clusters & Noise
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    unique_clusters = sorted(set(labels))
    for cid in unique_clusters:
        mask = labels == cid
        if cid == -1:
            axes[1].scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=PINK, s=24, alpha=0.85, marker="x", label=f"Noise (-1) (n={int(mask.sum())})"
            )
        else:
            axes[1].scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=PALETTE[int(cid) % len(PALETTE)], s=13, alpha=0.5,
                label=f"Cluster {cid} (n={int(mask.sum())})"
            )

    axes[1].set_title("DBSCAN Density Clusters & Noise in 2D PCA Space", fontsize=11, fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(CHART_DIR / "dbscan_clusters.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def get_dbscan_clustering():

    df, X = prepare_data()

    # DBSCAN parameters
    eps = 0.8
    min_samples = 5

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X)

    result_df = df.copy()
    result_df["Cluster"] = labels

    # Count clusters
    cluster_counts = (
        result_df["Cluster"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # Noise points are represented by -1
    noise_count = int((labels == -1).sum())

    # Valid clusters excluding noise
    valid_labels = set(labels)
    valid_labels.discard(-1)

    if len(valid_labels) >= 2:
        valid_mask = labels != -1
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]

        silhouette = silhouette_score(
            X_valid,
            labels_valid,
            sample_size=min(3000, len(X_valid)),
            random_state=42,
        )
        davies_bouldin = davies_bouldin_score(
            X_valid,
            labels_valid
        )
    else:
        silhouette = 0.0
        davies_bouldin = 0.0

    _generate_dbscan_charts(X, labels, eps, min_samples)

    return {
        "method": "DBSCAN",
        "algorithm": "Density-Based Spatial Clustering",
        "eps": eps,
        "min_samples": min_samples,
        "total_rows": len(df),
        "features": FEATURES,
        "cluster_count": len(valid_labels),
        "noise_count": noise_count,
        "silhouette_score": round(float(silhouette), 4),
        "davies_bouldin_score": round(float(davies_bouldin), 4),
        "dbscan_chart": "charts/dbscan_clusters.png",
        "clusters": {
            str(key): int(value)
            for key, value in cluster_counts.items()
        }
    }


# Compatibility function
def dbscan_clustering():

    return get_dbscan_clustering()