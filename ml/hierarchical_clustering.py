from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
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


def prepare_data():

    df = load_data()

    X = df[FEATURES].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return df, X_scaled


def _generate_dendrogram_charts(df, X):

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Representative sample for fast, crisp SciPy hierarchical linkage & cophenetic correlation
    rng = np.random.RandomState(42)
    sample_n = min(1200, len(X))
    sample_idx = rng.choice(len(X), size=sample_n, replace=False)
    X_sample = X[sample_idx]
    dist_vec = pdist(X_sample)

    # 1. Main Ward Linkage Dendrogram with Threshold Cut Line
    Z_ward = linkage(X_sample, method="ward")
    coph_ward, _ = cophenet(Z_ward, dist_vec)

    # Cut threshold between the last merge (1 cluster) and second-to-last merge (2 clusters)
    cut_height_k2 = float((Z_ward[-1, 2] + Z_ward[-2, 2]) / 2.0)
    cut_height_k3 = float((Z_ward[-2, 2] + Z_ward[-3, 2]) / 2.0)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=110)
    dendrogram(
        Z_ward,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=45.0,
        leaf_font_size=9.0,
        show_contracted=True,
        color_threshold=cut_height_k2,
        above_threshold_color="#948da7",
        ax=ax,
    )
    ax.axhline(
        y=cut_height_k2,
        color=PINK,
        linestyle="--",
        linewidth=1.8,
        label=f"Cut for K=2 Clusters (height = {cut_height_k2:.1f})",
    )
    ax.axhline(
        y=cut_height_k3,
        color=GREEN,
        linestyle=":",
        linewidth=1.5,
        label=f"Cut for K=3 Clusters (height = {cut_height_k3:.1f})",
    )
    ax.set_title(
        "Hierarchical Clustering Dendrogram (Ward Linkage · Top 30 Merged Leaves)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Cluster Size (number of records in leaf node) or Sample Index")
    ax.set_ylabel("Ward Linkage Euclidean Distance")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    dendrogram_path = CHART_DIR / "hierarchical_dendrogram.png"
    fig.savefig(dendrogram_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 2. Multi-Linkage Dendrogram Comparison (Ward, Complete, Average, Single)
    linkage_methods = [
        ("ward", "Ward Linkage (Minimises Variance)", PURPLE),
        ("complete", "Complete Linkage (Max Distance)", PINK),
        ("average", "Average Linkage (UPGMA)", GREEN),
        ("single", "Single Linkage (Min Distance)", AMBER),
    ]

    fig2, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=110)
    coph_scores = {}

    for ax_sub, (method_name, title, color) in zip(axes.ravel(), linkage_methods):
        Z_m = linkage(X_sample, method=method_name)
        c_score, _ = cophenet(Z_m, dist_vec)
        coph_scores[method_name] = round(float(c_score), 4)
        cut_m = float((Z_m[-1, 2] + Z_m[-2, 2]) / 2.0)

        dendrogram(
            Z_m,
            truncate_mode="lastp",
            p=20,
            leaf_rotation=45.0,
            leaf_font_size=8.0,
            show_contracted=True,
            color_threshold=cut_m,
            above_threshold_color="#948da7",
            ax=ax_sub,
        )
        ax_sub.axhline(y=cut_m, color=color, linestyle="--", linewidth=1.4, label=f"K=2 Cut ({cut_m:.1f})")
        ax_sub.set_title(f"{title} · Cophenetic r = {c_score:.3f}", fontsize=10.5, fontweight="bold")
        ax_sub.set_xlabel("Merged Subtrees (Top 20)")
        ax_sub.set_ylabel("Linkage Distance")
        ax_sub.legend(loc="upper right", fontsize=8)
        ax_sub.grid(axis="y", linestyle="--", alpha=0.3)

    fig2.suptitle("Dendrogram Comparison Across Linkage Criteria", fontsize=13, fontweight="bold", y=1.01)
    fig2.tight_layout()
    linkages_path = CHART_DIR / "hierarchical_linkages.png"
    fig2.savefig(linkages_path, dpi=110, bbox_inches="tight")
    plt.close(fig2)

    return coph_scores, round(cut_height_k2, 2), round(cut_height_k3, 2)


def _generate_cluster_scatter(df, X, labels):

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=110)

    # Left: PCA 2D Projection of Hierarchical Clusters
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        color = PALETTE[int(cluster_id) % len(PALETTE)]
        axes[0].scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=14,
            alpha=0.55,
            color=color,
            label=f"Cluster {cluster_id} (n={int(mask.sum())})",
        )
        centroid = X_pca[mask].mean(axis=0)
        axes[0].scatter(
            centroid[0],
            centroid[1],
            s=160,
            color=color,
            edgecolor="#33303e",
            linewidth=1.8,
            marker="X",
            zorder=5,
        )

    axes[0].set_title("Agglomerative Clusters in 2D PCA Space", fontsize=11, fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Right: Environmental Space (Light vs CO2)
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        color = PALETTE[int(cluster_id) % len(PALETTE)]
        axes[1].scatter(
            df.loc[mask, "Light (Fux)"],
            df.loc[mask, "CO2 (pmm)"],
            s=14,
            alpha=0.55,
            color=color,
            label=f"Cluster {cluster_id}",
        )

    axes[1].set_title("Hierarchical Clusters: Light (Fux) vs CO2 (pmm)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Light (Fux)")
    axes[1].set_ylabel("CO2 (pmm)")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    scatter_path = CHART_DIR / "hierarchical_clusters.png"
    fig.savefig(scatter_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def get_hierarchical_clustering():

    df, X = prepare_data()

    # Default number of clusters
    n_clusters = 2

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )

    labels = model.fit_predict(X)

    result_df = df.copy()
    result_df["Cluster"] = labels

    unique_labels = sorted(result_df["Cluster"].unique())

    cluster_counts = (
        result_df["Cluster"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if len(unique_labels) > 1:
        silhouette = silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=42)
        davies_bouldin = davies_bouldin_score(X, labels)
    else:
        silhouette = 0.0
        davies_bouldin = 0.0

    # Generate Dendrograms and Cluster Scatter plots
    coph_scores, cut_k2, cut_k3 = _generate_dendrogram_charts(df, X)
    _generate_cluster_scatter(df, X, labels)

    # Compare linkage methods on a consistent subset for speed & accuracy
    rng = np.random.RandomState(42)
    eval_idx = rng.choice(len(X), size=min(2500, len(X)), replace=False)
    X_eval = X[eval_idx]

    linkage_comparison = []
    for method_name in ("ward", "complete", "average", "single"):
        m_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage=method_name).fit_predict(X_eval)
        if len(set(m_labels)) > 1:
            sil_m = round(float(silhouette_score(X_eval, m_labels)), 4)
            db_m = round(float(davies_bouldin_score(X_eval, m_labels)), 4)
        else:
            sil_m = 0.0
            db_m = 0.0
        counts_m = pd.Series(m_labels).value_counts().sort_index().to_dict()
        linkage_comparison.append({
            "linkage": method_name.capitalize(),
            "silhouette": sil_m,
            "davies_bouldin": db_m,
            "cophenetic_corr": coph_scores.get(method_name, 0.0),
            "cluster_0": int(counts_m.get(0, 0)),
            "cluster_1": int(counts_m.get(1, 0)),
        })

    # Cluster feature means
    cluster_profiles = []
    for cid in unique_labels:
        sub = result_df[result_df["Cluster"] == cid]
        cluster_profiles.append({
            "cluster": f"Cluster {cid}",
            "count": int(len(sub)),
            "share": round(len(sub) / len(result_df) * 100, 1),
            "temp": round(float(sub["Temp"].mean()), 2),
            "humid": round(float(sub["Humid (%)"].mean()), 2),
            "light": round(float(sub["Light (Fux)"].mean()), 2),
            "co2": round(float(sub["CO2 (pmm)"].mean()), 2),
        })

    return {
        "method": "Hierarchical Clustering",
        "algorithm": "Agglomerative Clustering",
        "linkage": "Ward",
        "n_clusters": n_clusters,
        "total_rows": len(df),
        "features": FEATURES,
        "silhouette_score": round(float(silhouette), 4),
        "davies_bouldin_score": round(float(davies_bouldin), 4),
        "cophenetic_corr": coph_scores.get("ward", 0.0),
        "cut_height_k2": cut_k2,
        "cut_height_k3": cut_k3,
        "dendrogram_chart": "charts/hierarchical_dendrogram.png",
        "linkages_chart": "charts/hierarchical_linkages.png",
        "clusters_chart": "charts/hierarchical_clusters.png",
        "linkage_comparison": linkage_comparison,
        "cluster_profiles": cluster_profiles,
        "clusters": {
            str(key): int(value)
            for key, value in cluster_counts.items()
        }
    }


# Compatibility function
def hierarchical_clustering():

    return get_hierarchical_clustering()