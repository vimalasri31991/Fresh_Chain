from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

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


def get_anomaly_detection():

    df = load_data().copy()
    df["Class"] = (
        df["Class"].astype(str).str.strip().str.lower()
        .map({"bad": "BAD", "good": "Good"})
    )

    X_raw = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    contamination = 0.05

    # ============================================================
    # 1. ISOLATION FOREST (Primary Model: Before vs After Tuning)
    # ============================================================
    iso_before = IsolationForest(
        n_estimators=50,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )
    pred_before = iso_before.fit_predict(X_scaled)
    scores_before = iso_before.decision_function(X_scaled)

    iso_model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso_pred = iso_model.fit_predict(X_scaled)       # 1 = Normal, -1 = Anomaly
    iso_scores = iso_model.decision_function(X_scaled)  # negative = anomaly

    # ============================================================
    # 2. LOCAL OUTLIER FACTOR (Density-Based Anomaly Detection)
    # ============================================================
    lof_model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        n_jobs=-1,
    )
    lof_pred = lof_model.fit_predict(X_scaled)
    lof_scores = lof_model.negative_outlier_factor_

    # ============================================================
    # 3. ONE-CLASS SVM (Boundary-Based Anomaly Detection)
    #    Fit on representative sample for speed, predict all rows
    # ============================================================
    rng = np.random.RandomState(42)
    svm_train_idx = rng.choice(len(X_scaled), size=min(2500, len(X_scaled)), replace=False)
    ocsvm_model = OneClassSVM(kernel="rbf", nu=contamination, gamma="scale")
    ocsvm_model.fit(X_scaled[svm_train_idx])
    ocsvm_pred = ocsvm_model.predict(X_scaled)

    # ============================================================
    # 4. ELLIPTIC ENVELOPE (Gaussian / Mahalanobis Distance)
    # ============================================================
    elliptic_model = EllipticEnvelope(contamination=contamination, support_fraction=0.75, random_state=42)
    elliptic_pred = elliptic_model.fit_predict(X_scaled)

    # Consensus count: how many algorithms flagged each row as an anomaly (-1)
    anomaly_votes = (
        (iso_pred == -1).astype(int)
        + (lof_pred == -1).astype(int)
        + (ocsvm_pred == -1).astype(int)
        + (elliptic_pred == -1).astype(int)
    )

    df["Anomaly_Label"] = np.where(iso_pred == -1, "Anomaly", "Normal")
    df["Anomaly_Score"] = np.round(iso_scores, 4)
    df["Votes"] = anomaly_votes

    n_total = len(df)
    n_anomalies = int((iso_pred == -1).sum())
    n_normal = int((iso_pred == 1).sum())
    consensus_anomalies = int((anomaly_votes >= 2).sum())
    high_conf_anomalies = int((anomaly_votes >= 3).sum())

    # Separation quality metric: Normalized Mean Score Gap (Inliers vs Outliers)
    before_inlier_gap = float(scores_before[pred_before == 1].mean() - scores_before[pred_before == -1].mean())
    after_inlier_gap = float(iso_scores[iso_pred == 1].mean() - iso_scores[iso_pred == -1].mean())
    inlier_purity = round(n_normal / n_total * 100, 2)

    # Method comparison summary
    methods_summary = [
        {
            "name": "Isolation Forest",
            "type": "Tree Isolation",
            "params": "n_estimators=200, contamination=0.05",
            "normal_count": int((iso_pred == 1).sum()),
            "anomaly_count": int((iso_pred == -1).sum()),
            "anomaly_pct": round(float((iso_pred == -1).mean() * 100), 2),
            "bad_class_overlap": round(float((df.loc[iso_pred == -1, "Class"] == "BAD").mean() * 100), 1),
        },
        {
            "name": "Local Outlier Factor (LOF)",
            "type": "Local Density",
            "params": "n_neighbors=20, contamination=0.05",
            "normal_count": int((lof_pred == 1).sum()),
            "anomaly_count": int((lof_pred == -1).sum()),
            "anomaly_pct": round(float((lof_pred == -1).mean() * 100), 2),
            "bad_class_overlap": round(float((df.loc[lof_pred == -1, "Class"] == "BAD").mean() * 100), 1),
        },
        {
            "name": "One-Class SVM",
            "type": "RBF Boundary",
            "params": "kernel='rbf', nu=0.05, gamma='scale'",
            "normal_count": int((ocsvm_pred == 1).sum()),
            "anomaly_count": int((ocsvm_pred == -1).sum()),
            "anomaly_pct": round(float((ocsvm_pred == -1).mean() * 100), 2),
            "bad_class_overlap": round(float((df.loc[ocsvm_pred == -1, "Class"] == "BAD").mean() * 100), 1),
        },
        {
            "name": "Elliptic Envelope",
            "type": "Mahalanobis Covariance",
            "params": "contamination=0.05",
            "normal_count": int((elliptic_pred == 1).sum()),
            "anomaly_count": int((elliptic_pred == -1).sum()),
            "anomaly_pct": round(float((elliptic_pred == -1).mean() * 100), 2),
            "bad_class_overlap": round(float((df.loc[elliptic_pred == -1, "Class"] == "BAD").mean() * 100), 1),
        },
    ]

    # Feature-wise Normal vs Anomaly Comparison
    feature_comparison = []
    for col in FEATURES:
        norm_vals = df.loc[iso_pred == 1, col]
        anom_vals = df.loc[iso_pred == -1, col]
        z_shift = abs(anom_vals.mean() - norm_vals.mean()) / (norm_vals.std() + 1e-9)
        feature_comparison.append({
            "feature": col,
            "normal_mean": round(float(norm_vals.mean()), 2),
            "normal_std": round(float(norm_vals.std()), 2),
            "anomaly_mean": round(float(anom_vals.mean()), 2),
            "anomaly_std": round(float(anom_vals.std()), 2),
            "anomaly_min": round(float(anom_vals.min()), 2),
            "anomaly_max": round(float(anom_vals.max()), 2),
            "z_shift": round(float(z_shift), 2),
        })

    # Fruit-wise anomaly distribution
    fruit_anomalies = []
    for fruit_name, grp in df.groupby("Fruit"):
        anom_c = int((grp["Anomaly_Label"] == "Anomaly").sum())
        fruit_anomalies.append({
            "fruit": fruit_name,
            "total": int(len(grp)),
            "anomalies": anom_c,
            "rate": round(anom_c / len(grp) * 100, 2),
        })
    fruit_anomalies.sort(key=lambda r: r["anomalies"], reverse=True)

    # Top 10 most severe anomalies (lowest decision_function score)
    top_df = df.sort_values("Anomaly_Score", ascending=True).head(10)
    top_anomalies = [
        {
            "rank": idx + 1,
            "row_id": int(row_idx),
            "fruit": row["Fruit"],
            "temp": round(float(row["Temp"]), 2),
            "humid": round(float(row["Humid (%)"]), 2),
            "light": round(float(row["Light (Fux)"]), 2),
            "co2": round(float(row["CO2 (pmm)"]), 2),
            "quality_class": row["Class"],
            "score": round(float(row["Anomaly_Score"]), 4),
            "votes": int(row["Votes"]),
        }
        for idx, (row_idx, row) in enumerate(top_df.iterrows())
    ]

    # ============================================================
    # CHARTS
    # ============================================================
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X_scaled)

    # Chart 1: Isolation Forest Decision Boundary & 2D Scatter
    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.6), dpi=110)

    iso_2d = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
    iso_2d.fit(X_pca)
    xx, yy = np.meshgrid(
        np.linspace(X_pca[:, 0].min() - 0.6, X_pca[:, 0].max() + 0.6, 120),
        np.linspace(X_pca[:, 1].min() - 0.6, X_pca[:, 1].max() + 0.6, 120),
    )
    Z_grid = iso_2d.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes1[0].contourf(xx, yy, Z_grid, levels=np.linspace(Z_grid.min(), 0, 7), cmap="PuRd_r", alpha=0.22)
    axes1[0].contour(xx, yy, Z_grid, levels=[0], linewidths=1.8, colors=[PINK])
    axes1[0].scatter(
        X_pca[iso_pred == 1, 0], X_pca[iso_pred == 1, 1],
        c=GREEN, s=11, alpha=0.45, label=f"Normal Inliers (n={n_normal})"
    )
    axes1[0].scatter(
        X_pca[iso_pred == -1, 0], X_pca[iso_pred == -1, 1],
        c=PINK, s=26, alpha=0.85, marker="x", label=f"Anomalies (n={n_anomalies})"
    )
    axes1[0].set_title("Isolation Forest Decision Boundary (PCA Plane)", fontsize=11, fontweight="bold")
    axes1[0].set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var)")
    axes1[0].set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var)")
    axes1[0].legend(loc="upper right", fontsize=8.5)
    axes1[0].grid(True, linestyle="--", alpha=0.3)

    axes1[1].scatter(
        df.loc[iso_pred == 1, "Light (Fux)"], df.loc[iso_pred == 1, "CO2 (pmm)"],
        c=GREEN, s=12, alpha=0.45, label="Normal Readings"
    )
    axes1[1].scatter(
        df.loc[iso_pred == -1, "Light (Fux)"], df.loc[iso_pred == -1, "CO2 (pmm)"],
        c=PINK, s=28, alpha=0.85, marker="x", label="Detected Anomalies"
    )
    axes1[1].set_title("Environmental Sensor Anomalies: Light vs CO2", fontsize=11, fontweight="bold")
    axes1[1].set_xlabel("Light (Fux)")
    axes1[1].set_ylabel("CO2 (pmm)")
    axes1[1].legend(loc="upper right", fontsize=8.5)
    axes1[1].grid(True, linestyle="--", alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(CHART_DIR / "anomaly_isolation_forest.png", dpi=110, bbox_inches="tight")
    plt.close(fig1)

    # Chart 2: Anomaly Score Distribution & Consensus Breakdown
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.4), dpi=110)

    axes2[0].hist(
        iso_scores[iso_pred == 1], bins=35, color=GREEN, alpha=0.78,
        edgecolor="white", label="Normal Inliers (Score >= 0)"
    )
    axes2[0].hist(
        iso_scores[iso_pred == -1], bins=20, color=PINK, alpha=0.85,
        edgecolor="white", label="Anomalies (Score < 0)"
    )
    axes2[0].axvline(0.0, color="#33303e", linestyle="--", linewidth=1.8, label="Decision Threshold (0.0)")
    axes2[0].set_title("Isolation Forest Anomaly Score Distribution", fontsize=11, fontweight="bold")
    axes2[0].set_xlabel("Anomaly Decision Score (Lower = More Anomalous)")
    axes2[0].set_ylabel("Number of Records")
    axes2[0].legend(fontsize=8.5)
    axes2[0].grid(axis="y", linestyle="--", alpha=0.3)

    vote_counts = pd.Series(anomaly_votes).value_counts().reindex([0, 1, 2, 3, 4], fill_value=0)
    bar_colors = [GREEN, AMBER, PURPLE, PINK, "#a34a3a"]
    bars = axes2[1].bar(
        ["0 (Normal)", "1 Model", "2 Models", "3 Models", "All 4 Models"],
        vote_counts.values,
        color=bar_colors,
        edgecolor="white",
        width=0.6,
    )
    for b in bars:
        h = int(b.get_height())
        axes2[1].annotate(
            f"{h}",
            (b.get_x() + b.get_width() / 2, h),
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    axes2[1].set_title("Multi-Algorithm Anomaly Consensus Votes", fontsize=11, fontweight="bold")
    axes2[1].set_xlabel("Number of Algorithms Flagging Record as Anomaly")
    axes2[1].set_ylabel("Record Count")
    axes2[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(CHART_DIR / "anomaly_score_distribution.png", dpi=110, bbox_inches="tight")
    plt.close(fig2)

    # Chart 3: Side-by-side Comparison of 4 Anomaly Detection Algorithms
    fig3, axes3 = plt.subplots(2, 2, figsize=(11, 7.2), dpi=110)
    algo_list = [
        ("Isolation Forest", iso_pred),
        ("Local Outlier Factor (LOF)", lof_pred),
        ("One-Class SVM (RBF)", ocsvm_pred),
        ("Elliptic Envelope (Mahalanobis)", elliptic_pred),
    ]
    for ax_sub, (title, preds) in zip(axes3.ravel(), algo_list):
        ax_sub.scatter(
            X_pca[preds == 1, 0], X_pca[preds == 1, 1],
            c="#b3abc4", s=10, alpha=0.4, label=f"Normal ({int((preds == 1).sum())})"
        )
        ax_sub.scatter(
            X_pca[preds == -1, 0], X_pca[preds == -1, 1],
            c=PINK, s=22, alpha=0.85, marker="x", label=f"Anomaly ({int((preds == -1).sum())})"
        )
        ax_sub.set_title(title, fontsize=10.5, fontweight="bold")
        ax_sub.set_xlabel("PC1")
        ax_sub.set_ylabel("PC2")
        ax_sub.legend(loc="upper right", fontsize=8)
        ax_sub.grid(True, linestyle="--", alpha=0.3)

    fig3.suptitle("Anomaly Detection Comparison Across 4 Unsupervised Models", fontsize=12.5, fontweight="bold", y=1.01)
    fig3.tight_layout()
    fig3.savefig(CHART_DIR / "anomaly_methods_comparison.png", dpi=110, bbox_inches="tight")
    plt.close(fig3)

    # Chart 4: Feature Distributions Normal vs Anomaly
    fig4, axes4 = plt.subplots(2, 2, figsize=(10.5, 6.2), dpi=110)
    for ax_sub, col in zip(axes4.ravel(), FEATURES):
        norm_data = df.loc[iso_pred == 1, col]
        anom_data = df.loc[iso_pred == -1, col]
        bp = ax_sub.boxplot(
            [norm_data, anom_data],
            tick_labels=["Normal Inliers", "Detected Anomalies"],
            patch_artist=True,
            widths=0.45,
        )
        bp["boxes"][0].set_facecolor("#e3f2e8")
        bp["boxes"][0].set_edgecolor(GREEN)
        bp["boxes"][1].set_facecolor("#fbe4ec")
        bp["boxes"][1].set_edgecolor(PINK)
        ax_sub.set_title(f"{col}: Normal vs Anomaly", fontsize=10.5, fontweight="bold")
        ax_sub.set_ylabel(col)
        ax_sub.grid(axis="y", linestyle="--", alpha=0.3)

    fig4.suptitle("Environmental Feature Spread: Normal vs Anomalous Records", fontsize=12.5, fontweight="bold", y=1.01)
    fig4.tight_layout()
    fig4.savefig(CHART_DIR / "anomaly_feature_boxplots.png", dpi=110, bbox_inches="tight")
    plt.close(fig4)

    return {
        "method": "Anomaly Detection",
        "primary_algorithm": "Isolation Forest",
        "total_rows": n_total,
        "normal_count": n_normal,
        "anomaly_count": n_anomalies,
        "contamination": contamination,
        "contamination_pct": round(contamination * 100, 1),
        "inlier_purity": inlier_purity,
        "consensus_anomalies": consensus_anomalies,
        "high_conf_anomalies": high_conf_anomalies,
        "before_score_gap": round(before_inlier_gap * 100, 2),
        "after_score_gap": round(after_inlier_gap * 100, 2),
        "features": FEATURES,
        "methods_summary": methods_summary,
        "feature_comparison": feature_comparison,
        "fruit_anomalies": fruit_anomalies,
        "top_anomalies": top_anomalies,
        "isolation_chart": "charts/anomaly_isolation_forest.png",
        "score_chart": "charts/anomaly_score_distribution.png",
        "methods_chart": "charts/anomaly_methods_comparison.png",
        "boxplot_chart": "charts/anomaly_feature_boxplots.png",
    }
