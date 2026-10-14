from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_score
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


def get_pca_analysis():

    df = load_data().copy()
    df["Class"] = (
        df["Class"].astype(str).str.strip().str.lower()
        .map({"bad": "BAD", "good": "Good"})
    )

    X_raw = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Full PCA across all 4 environmental dimensions
    pca_full = PCA(n_components=len(FEATURES), random_state=42)
    X_pca_full = pca_full.fit_transform(X_scaled)

    # 2D & 3D PCA for dimensionality reduction & reconstruction error
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    X_rec_2d = pca_2d.inverse_transform(X_pca_2d)
    recon_mse_2d = round(float(mean_squared_error(X_scaled, X_rec_2d)), 4)

    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    X_rec_3d = pca_3d.inverse_transform(X_pca_3d)
    recon_mse_3d = round(float(mean_squared_error(X_scaled, X_rec_3d)), 4)

    explained_ratios = [round(float(v * 100), 2) for v in pca_full.explained_variance_ratio_]
    eigenvalues = [round(float(v), 4) for v in pca_full.explained_variance_]
    cumulative_ratios = [round(float(v * 100), 2) for v in np.cumsum(pca_full.explained_variance_ratio_)]

    components_summary = []
    for idx in range(len(FEATURES)):
        components_summary.append({
            "component": f"PC{idx + 1}",
            "eigenvalue": eigenvalues[idx],
            "variance_pct": explained_ratios[idx],
            "cumulative_pct": cumulative_ratios[idx],
            "singular_value": round(float(pca_full.singular_values_[idx]), 2),
        })

    # Loadings Matrix (components_ has shape [n_components, n_features])
    loadings_rows = []
    for f_idx, feat_name in enumerate(FEATURES):
        pc_vals = [round(float(pca_full.components_[c_idx, f_idx]), 4) for c_idx in range(len(FEATURES))]
        dominant_pc = f"PC{int(np.argmax(np.abs(pc_vals))) + 1}"
        loadings_rows.append({
            "feature": feat_name,
            "pc1": pc_vals[0],
            "pc2": pc_vals[1],
            "pc3": pc_vals[2],
            "pc4": pc_vals[3],
            "dominant": dominant_pc,
        })

    # Sample of projected records
    preview_rows = []
    for i in range(min(8, len(df))):
        preview_rows.append({
            "id": i + 1,
            "fruit": df.iloc[i]["Fruit"],
            "quality_class": df.iloc[i]["Class"],
            "pc1": round(float(X_pca_full[i, 0]), 4),
            "pc2": round(float(X_pca_full[i, 1]), 4),
            "pc3": round(float(X_pca_full[i, 2]), 4),
            "pc4": round(float(X_pca_full[i, 3]), 4),
        })

    # Class separation in 2D PCA space
    y_binary = (df["Class"] == "Good").astype(int).values
    pca_silhouette = round(
        float(silhouette_score(X_pca_2d, y_binary, sample_size=min(2500, len(X_pca_2d)), random_state=42)),
        4,
    )

    # ============================================================
    # CHARTS
    # ============================================================
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 1: Scree Plot (Individual + Cumulative Explained Variance)
    fig1, ax1 = plt.subplots(figsize=(8.8, 4.6), dpi=110)
    pcs = [f"PC{i+1}" for i in range(len(FEATURES))]
    bars = ax1.bar(pcs, explained_ratios, color=PURPLE, alpha=0.82, width=0.5, label="Individual Explained Variance (%)")
    ax1.plot(pcs, cumulative_ratios, color=PINK, marker="o", linewidth=2.4, markersize=7, label="Cumulative Variance (%)")
    ax1.axhline(80.0, color=GREEN, linestyle="--", linewidth=1.4, label="80% Variance Threshold")

    for b, cum in zip(bars, cumulative_ratios):
        h = b.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            (b.get_x() + b.get_width() / 2, h / 2),
            ha="center", va="center", color="white", fontweight="bold", fontsize=9.5
        )
    for idx, cum in enumerate(cumulative_ratios):
        ax1.annotate(
            f"Cum: {cum:.1f}%",
            (idx, cum + 2.5),
            ha="center", va="bottom", color=PINK, fontweight="bold", fontsize=9
        )

    ax1.set_ylim(0, 112)
    ax1.set_title("PCA Scree Plot — Explained & Cumulative Variance by Principal Component", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Principal Components")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.legend(loc="center right", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    fig1.tight_layout()
    fig1.savefig(CHART_DIR / "pca_scree_plot.png", dpi=110, bbox_inches="tight")
    plt.close(fig1)

    # Chart 2: 2D PCA Projection (Colored by Quality Class & by Fruit Type)
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.7), dpi=110)

    for cls_name in ["BAD", "Good"]:
        mask = df["Class"] == cls_name
        axes2[0].scatter(
            X_pca_2d[mask, 0], X_pca_2d[mask, 1],
            c=CLASS_COLORS[cls_name], s=13, alpha=0.5, label=f"{cls_name} (n={int(mask.sum())})"
        )
    axes2[0].set_title("2D PCA Projection by Quality Class", fontsize=11, fontweight="bold")
    axes2[0].set_xlabel(f"PC1 ({explained_ratios[0]:.1f}% variance)")
    axes2[0].set_ylabel(f"PC2 ({explained_ratios[1]:.1f}% variance)")
    axes2[0].legend(fontsize=8.5)
    axes2[0].grid(True, linestyle="--", alpha=0.3)

    for f_idx, fruit_name in enumerate(sorted(df["Fruit"].unique())):
        mask = df["Fruit"] == fruit_name
        axes2[1].scatter(
            X_pca_2d[mask, 0], X_pca_2d[mask, 1],
            c=PALETTE[f_idx % len(PALETTE)], s=13, alpha=0.5, label=fruit_name
        )
    axes2[1].set_title("2D PCA Projection by Fruit Type", fontsize=11, fontweight="bold")
    axes2[1].set_xlabel(f"PC1 ({explained_ratios[0]:.1f}% variance)")
    axes2[1].set_ylabel(f"PC2 ({explained_ratios[1]:.1f}% variance)")
    axes2[1].legend(fontsize=8.5)
    axes2[1].grid(True, linestyle="--", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(CHART_DIR / "pca_2d_scatter.png", dpi=110, bbox_inches="tight")
    plt.close(fig2)

    # Chart 3: PCA Biplot (Feature Vectors) + Loadings Heatmap
    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.7), dpi=110)

    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(X_pca_2d), size=min(1500, len(X_pca_2d)), replace=False)
    axes3[0].scatter(
        X_pca_2d[sub_idx, 0], X_pca_2d[sub_idx, 1],
        c="#d4cce3", s=10, alpha=0.45, label="Projected Records"
    )
    scale_factor = np.max(np.abs(X_pca_2d[sub_idx])) * 0.78
    for f_idx, feat_name in enumerate(FEATURES):
        vx = pca_full.components_[0, f_idx] * scale_factor
        vy = pca_full.components_[1, f_idx] * scale_factor
        axes3[0].annotate(
            "",
            xy=(vx, vy),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=PINK, lw=2.2),
        )
        axes3[0].text(
            vx * 1.12, vy * 1.12, feat_name,
            color="#33303e", fontweight="bold", fontsize=9.5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#fdfcfa", ec="#ece2f7", alpha=0.9)
        )
    axes3[0].axhline(0, color="#948da7", linestyle=":", linewidth=1)
    axes3[0].axvline(0, color="#948da7", linestyle=":", linewidth=1)
    axes3[0].set_title("PCA Biplot — Feature Loading Vectors on PC1 vs PC2", fontsize=11, fontweight="bold")
    axes3[0].set_xlabel(f"PC1 ({explained_ratios[0]:.1f}%)")
    axes3[0].set_ylabel(f"PC2 ({explained_ratios[1]:.1f}%)")
    axes3[0].grid(True, linestyle="--", alpha=0.3)

    loadings_df = pd.DataFrame(
        pca_full.components_.T,
        index=FEATURES,
        columns=[f"PC{i+1}" for i in range(len(FEATURES))],
    )
    sns.heatmap(
        loadings_df, annot=True, cmap="PuOr", fmt=".2f",
        vmin=-1, vmax=1, cbar=True, ax=axes3[1]
    )
    axes3[1].set_title("Principal Component Feature Loadings Matrix", fontsize=11, fontweight="bold")

    fig3.tight_layout()
    fig3.savefig(CHART_DIR / "pca_biplot_loadings.png", dpi=110, bbox_inches="tight")
    plt.close(fig3)

    return {
        "method": "Principal Component Analysis (PCA)",
        "total_rows": len(df),
        "n_features": len(FEATURES),
        "n_components": len(FEATURES),
        "features": FEATURES,
        "pc1_variance": explained_ratios[0],
        "pc2_variance": explained_ratios[1],
        "pc3_variance": explained_ratios[2],
        "pc4_variance": explained_ratios[3],
        "variance_2d": cumulative_ratios[1],
        "variance_3d": cumulative_ratios[2],
        "recon_mse_2d": recon_mse_2d,
        "recon_mse_3d": recon_mse_3d,
        "pca_silhouette": pca_silhouette,
        "components_summary": components_summary,
        "loadings_rows": loadings_rows,
        "preview_rows": preview_rows,
        "scree_chart": "charts/pca_scree_plot.png",
        "scatter_chart": "charts/pca_2d_scatter.png",
        "biplot_chart": "charts/pca_biplot_loadings.png",
    }
