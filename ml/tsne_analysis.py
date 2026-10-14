from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
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


def get_tsne_analysis():

    df = load_data().copy()
    df["Class"] = (
        df["Class"].astype(str).str.strip().str.lower()
        .map({"bad": "BAD", "good": "Good"})
    )

    # Stratified sample of 1,500 records for fast, high-resolution manifold embedding
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

    # Compare Perplexities: 10 (local), 30 (balanced optimal), 50 (broad global)
    perplexities = [10, 30, 50]
    embeddings = {}
    perplexity_comparison = []

    y_binary = (df_sample["Class"] == "Good").astype(int).values

    for perp in perplexities:
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate="auto",
            init="pca",
            max_iter=750,
            random_state=42,
            n_jobs=-1,
        )
        emb = tsne.fit_transform(X_scaled)
        embeddings[perp] = emb

        tw = round(float(trustworthiness(X_scaled, emb, n_neighbors=10)), 4)
        sil = round(float(silhouette_score(emb, y_binary)), 4)
        kl_div = round(float(tsne.kl_divergence_), 4)

        perplexity_comparison.append({
            "perplexity": perp,
            "kl_divergence": kl_div,
            "trustworthiness": tw,
            "trustworthiness_pct": round(tw * 100, 2),
            "silhouette": sil,
            "iterations": int(tsne.n_iter_),
        })

    # Primary embedding: Perplexity = 30
    best_perp = 30
    X_tsne = embeddings[best_perp]
    primary_stats = next(r for r in perplexity_comparison if r["perplexity"] == best_perp)
    baseline_stats = next(r for r in perplexity_comparison if r["perplexity"] == 10)

    # Sample coordinate preview
    preview_rows = []
    for i in range(min(8, len(df_sample))):
        preview_rows.append({
            "id": i + 1,
            "fruit": df_sample.iloc[i]["Fruit"],
            "quality_class": df_sample.iloc[i]["Class"],
            "tsne_1": round(float(X_tsne[i, 0]), 3),
            "tsne_2": round(float(X_tsne[i, 1]), 3),
        })

    # ============================================================
    # CHARTS
    # ============================================================
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 1: 2D t-SNE Manifold by Quality Class and Fruit Type
    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.7), dpi=110)

    for cls_name in ["BAD", "Good"]:
        mask = df_sample["Class"] == cls_name
        axes1[0].scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=CLASS_COLORS[cls_name], s=15, alpha=0.62, label=f"{cls_name} (n={int(mask.sum())})"
        )
    axes1[0].set_title(
        f"2D t-SNE Embedding by Quality Class (Perplexity={best_perp})",
        fontsize=11, fontweight="bold"
    )
    axes1[0].set_xlabel("t-SNE Dimension 1")
    axes1[0].set_ylabel("t-SNE Dimension 2")
    axes1[0].legend(fontsize=8.5)
    axes1[0].grid(True, linestyle="--", alpha=0.3)

    for f_idx, fruit_name in enumerate(sorted(df_sample["Fruit"].unique())):
        mask = df_sample["Fruit"] == fruit_name
        axes1[1].scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=PALETTE[f_idx % len(PALETTE)], s=15, alpha=0.62, label=fruit_name
        )
    axes1[1].set_title(
        f"2D t-SNE Embedding by Fruit Category (Perplexity={best_perp})",
        fontsize=11, fontweight="bold"
    )
    axes1[1].set_xlabel("t-SNE Dimension 1")
    axes1[1].set_ylabel("t-SNE Dimension 2")
    axes1[1].legend(fontsize=8.5)
    axes1[1].grid(True, linestyle="--", alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(CHART_DIR / "tsne_2d_scatter.png", dpi=110, bbox_inches="tight")
    plt.close(fig1)

    # Chart 2: Perplexity Comparison (Perplexity = 10, 30, 50)
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4.2), dpi=110)
    for ax_sub, stats in zip(axes2, perplexity_comparison):
        perp = stats["perplexity"]
        emb = embeddings[perp]
        for cls_name in ["BAD", "Good"]:
            mask = df_sample["Class"] == cls_name
            ax_sub.scatter(
                emb[mask, 0], emb[mask, 1],
                c=CLASS_COLORS[cls_name], s=11, alpha=0.58, label=cls_name
            )
        ax_sub.set_title(
            f"Perplexity = {perp}\nKL={stats['kl_divergence']:.2f} · Trust={stats['trustworthiness_pct']:.1f}%",
            fontsize=10, fontweight="bold"
        )
        ax_sub.set_xlabel("t-SNE 1")
        ax_sub.set_ylabel("t-SNE 2")
        ax_sub.legend(loc="upper right", fontsize=8)
        ax_sub.grid(True, linestyle="--", alpha=0.3)

    fig2.suptitle("Effect of Perplexity on t-SNE Manifold Topology", fontsize=12, fontweight="bold", y=1.02)
    fig2.tight_layout()
    fig2.savefig(CHART_DIR / "tsne_perplexity_comparison.png", dpi=110, bbox_inches="tight")
    plt.close(fig2)

    # Chart 3: t-SNE KDE Density Contour & KL Divergence Profile
    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.5), dpi=110)

    tsne_df = pd.DataFrame({
        "t-SNE 1": X_tsne[:, 0],
        "t-SNE 2": X_tsne[:, 1],
        "Class": df_sample["Class"].values,
    })
    sns.kdeplot(
        data=tsne_df, x="t-SNE 1", y="t-SNE 2", hue="Class",
        palette=CLASS_COLORS, fill=True, alpha=0.45, levels=7, ax=axes3[0]
    )
    axes3[0].set_title("t-SNE Manifold Density Contours by Quality Class", fontsize=11, fontweight="bold")
    axes3[0].grid(True, linestyle="--", alpha=0.3)

    perp_labels = [f"Perp = {r['perplexity']}" for r in perplexity_comparison]
    kl_vals = [r["kl_divergence"] for r in perplexity_comparison]
    tw_vals = [r["trustworthiness_pct"] for r in perplexity_comparison]

    x_pos = np.arange(len(perp_labels))
    ax_bar = axes3[1]
    bars = ax_bar.bar(x_pos, tw_vals, width=0.45, color=PURPLE, alpha=0.85, label="Trustworthiness (%)")
    for b in bars:
        h = b.get_height()
        ax_bar.annotate(
            f"{h:.1f}%",
            (b.get_x() + b.get_width() / 2, h - 6),
            ha="center", va="top", color="white", fontweight="bold", fontsize=9.5
        )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(perp_labels)
    ax_bar.set_ylim(0, 108)
    ax_bar.set_ylabel("Trustworthiness Score (%)")
    ax_bar.set_title("Manifold Trustworthiness Across Perplexity Settings", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(CHART_DIR / "tsne_density_contour.png", dpi=110, bbox_inches="tight")
    plt.close(fig3)

    return {
        "method": "t-SNE (t-Distributed Stochastic Neighbor Embedding)",
        "total_rows": len(df),
        "sample_rows": len(df_sample),
        "features": FEATURES,
        "n_components": 2,
        "perplexity": best_perp,
        "learning_rate": "auto",
        "init": "pca",
        "kl_divergence": primary_stats["kl_divergence"],
        "trustworthiness": primary_stats["trustworthiness"],
        "trustworthiness_pct": primary_stats["trustworthiness_pct"],
        "before_trustworthiness_pct": baseline_stats["trustworthiness_pct"],
        "silhouette_score": primary_stats["silhouette"],
        "perplexity_comparison": perplexity_comparison,
        "preview_rows": preview_rows,
        "scatter_chart": "charts/tsne_2d_scatter.png",
        "perplexity_chart": "charts/tsne_perplexity_comparison.png",
        "density_chart": "charts/tsne_density_contour.png",
    }
