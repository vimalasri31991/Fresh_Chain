import json
from pathlib import Path
from threading import Lock
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml.data_loader import load_preprocessed_data

BASE_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = BASE_DIR / "static" / "charts"

# FruitSense Pastel Palette
PURPLE = "#7c58c9"
PINK = "#c1477e"
GREEN = "#4c9a6c"
AMBER = "#c68a3d"
BLUE = "#3d79b8"
GREY = "#9c93b4"
LIGHT_BG = "#faf8f5"
BORDER_COLOR = "#ece2f7"

_CACHE = None
_LOCK = Lock()


# ============================================================
# HELPER: MATPLOTLIB PASTEL STYLING
# ============================================================
def _setup_figure(figsize=(11, 4.8), dpi=115):
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    for ax in (axes if isinstance(axes, np.ndarray) else [axes]):
        ax.set_facecolor("#fcfbfa")
        ax.grid(True, linestyle="--", alpha=0.35, color="#dcd3ea")
        for spine in ax.spines.values():
            spine.set_color("#ded8ea")
            spine.set_linewidth(1.0)
    return fig, axes


def _setup_single_figure(figsize=(9, 4.8), dpi=115):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fcfbfa")
    ax.grid(True, linestyle="--", alpha=0.35, color="#dcd3ea")
    for spine in ax.spines.values():
        spine.set_color("#ded8ea")
        spine.set_linewidth(1.0)
    return fig, ax


# ============================================================
# 1. TRAIN / VALIDATION / TEST SPLIT DISCIPLINE
# ============================================================
def _run_split_discipline(df):
    target = "Class_Encoded"
    features = [c for c in df.columns if c != target]

    X = df[features].copy()
    y = df[target].copy()

    total_samples = len(df)

    # 70% Train, 15% Validation, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Demonstrate Optimistic Bias: Tuning on Val vs Snooping Test Set
    depths = list(range(1, 25))
    val_scores = []
    test_scores = []
    train_scores = []

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
        val_scores.append(accuracy_score(y_val, clf.predict(X_val)))
        test_scores.append(accuracy_score(y_test, clf.predict(X_test)))

    best_val_idx = int(np.argmax(val_scores))
    best_val_depth = depths[best_val_idx]
    best_val_score = val_scores[best_val_idx]

    # Best model chosen strictly via validation set
    honest_model = DecisionTreeClassifier(max_depth=best_val_depth, random_state=42)
    honest_model.fit(X_train, y_train)
    honest_test_acc = accuracy_score(y_test, honest_model.predict(X_test))

    # Snooped model (violates discipline by tuning directly on test set)
    snooped_idx = int(np.argmax(test_scores))
    snooped_depth = depths[snooped_idx]
    snooped_test_acc = test_scores[snooped_idx]
    optimistic_bias = (snooped_test_acc - honest_test_acc) * 100

    # Chart generation
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = _setup_figure(figsize=(11, 4.6))

    # Left: Three-Way Partition Breakdown
    splits = ["Train Set\n(70%)", "Validation Set\n(15%)", "Held-Out Test Set\n(15%)"]
    counts = [len(X_train), len(X_val), len(X_test)]
    colors = [PURPLE, AMBER, GREEN]
    bars = axes[0].bar(splits, counts, color=colors, width=0.55, edgecolor="#ffffff", linewidth=1.5)
    for b, c in zip(bars, counts):
        h = b.get_height()
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            h / 2,
            f"{c} rows\n({c/total_samples*100:.1f}%)",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9.5,
        )
    axes[0].set_title("Partition Discipline — 70 / 15 / 15 Clean Isolation", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_ylim(0, len(X_train) * 1.18)

    # Right: Test Lockbox & Optimistic Bias Curve
    axes[1].plot(depths, [s * 100 for s in train_scores], color=GREY, linestyle=":", label="Training (Capacity)", linewidth=1.6)
    axes[1].plot(depths, [s * 100 for s in val_scores], color=AMBER, marker="s", markersize=4, label="Validation (Selection)", linewidth=2.0)
    axes[1].plot(depths, [s * 100 for s in test_scores], color=GREEN, marker="o", markersize=4, label="Held-Out Test (Evaluated Once)", linewidth=2.0)
    axes[1].scatter([best_val_depth], [honest_test_acc * 100], color=PURPLE, s=110, zorder=5, label=f"Honest Choice (d={best_val_depth})")
    axes[1].set_title("The Lockbox Rule — Preventing Hyperparameter Snooping", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Decision Tree Max Depth")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=8.5, loc="lower right")

    fig.tight_layout()
    chart_path = "eval_split_discipline.png"
    fig.savefig(CHART_DIR / chart_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "total_samples": total_samples,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "best_val_depth": best_val_depth,
        "best_val_score": round(best_val_score * 100, 2),
        "honest_test_acc": round(honest_test_acc * 100, 2),
        "snooped_depth": snooped_depth,
        "snooped_test_acc": round(snooped_test_acc * 100, 2),
        "optimistic_bias": round(optimistic_bias, 2),
        "chart": f"charts/{chart_path}",
    }


# ============================================================
# 2. K-FOLD, STRATIFIED K-FOLD & NESTED CV
# ============================================================
def _run_cross_validation(df):
    target = "Class_Encoded"
    features = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
    X = df[features].values
    y = df[target].values

    # Comparison: KFold vs StratifiedKFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    kf_positive_ratios = []
    skf_positive_ratios = []

    for train_idx, test_idx in kf.split(X, y):
        kf_positive_ratios.append(float(np.mean(y[test_idx]) * 100))

    for train_idx, test_idx in skf.split(X, y):
        skf_positive_ratios.append(float(np.mean(y[test_idx]) * 100))

    overall_pos_ratio = float(np.mean(y) * 100)
    kf_var = float(np.std(kf_positive_ratios))
    skf_var = float(np.std(skf_positive_ratios))

    # Nested Cross-Validation Demonstration
    # Outer loop (5 folds): Unbiased Generalisation Estimate
    # Inner loop (3 folds): Hyperparameter selection (GridSearch)
    param_grid = {"max_depth": [3, 6, 10, 15], "min_samples_leaf": [2, 10]}
    base_clf = DecisionTreeClassifier(random_state=42)

    # 1. Non-nested CV (Optimistically biased: tuning on all folds together)
    non_nested_gs = GridSearchCV(base_clf, param_grid, cv=5, scoring="accuracy")
    non_nested_gs.fit(X, y)
    non_nested_score = float(non_nested_gs.best_score_ * 100)

    # 2. Nested CV (Outer loop evaluates unbiased generalization)
    nested_scores = []
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]

        inner_gs = GridSearchCV(base_clf, param_grid, cv=inner_cv, scoring="accuracy")
        inner_gs.fit(X_outer_train, y_outer_train)

        best_model = inner_gs.best_estimator_
        outer_score = accuracy_score(y_outer_test, best_model.predict(X_outer_test))
        nested_scores.append(outer_score * 100)

    nested_mean = float(np.mean(nested_scores))
    nested_std = float(np.std(nested_scores))
    selection_bias = float(non_nested_score - nested_mean)

    # Chart generation
    fig, axes = _setup_figure(figsize=(11, 4.6))

    # Left: Fold Class Balance (Standard vs Stratified)
    fold_nums = np.arange(1, 6)
    width = 0.35
    axes[0].bar(fold_nums - width / 2, kf_positive_ratios, width, label=f"Standard K-Fold (σ={kf_var:.2f}%)", color=AMBER, alpha=0.85)
    axes[0].bar(fold_nums + width / 2, skf_positive_ratios, width, label=f"Stratified K-Fold (σ={skf_var:.2f}%)", color=PURPLE, alpha=0.95)
    axes[0].axhline(overall_pos_ratio, color=PINK, linestyle="--", linewidth=1.5, label=f"Dataset True Class Ratio ({overall_pos_ratio:.1f}%)")
    axes[0].set_title("Class Ratio Preservation Across 5 Folds", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Fold Index")
    axes[0].set_ylabel("Minority / Positive Class (%)")
    axes[0].set_ylim(overall_pos_ratio - 6, overall_pos_ratio + 6)
    axes[0].legend(fontsize=8, loc="lower right")

    # Right: Nested CV vs Non-Nested CV Score Comparison
    axes[1].plot(fold_nums, nested_scores, marker="o", color=PURPLE, linewidth=2, markersize=6, label=f"Nested Outer Folds (Mean={nested_mean:.2f}%)")
    axes[1].axhline(non_nested_score, color=PINK, linestyle="--", linewidth=1.8, label=f"Non-Nested CV Score ({non_nested_score:.2f}%) [Optimistic]")
    axes[1].axhline(nested_mean, color=GREEN, linestyle="-.", linewidth=1.8, label=f"Nested CV Unbiased Mean ({nested_mean:.2f}%)")
    axes[1].fill_between(fold_nums, nested_mean - nested_std, nested_mean + nested_std, color=PURPLE, alpha=0.12, label="Outer Folds ±1σ")
    axes[1].set_title(f"Nested CV — Removing Selection Bias (Bias: +{selection_bias:.2f} pts)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Outer Fold Index")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    chart_path = "eval_cross_validation.png"
    fig.savefig(CHART_DIR / chart_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_folds": 5,
        "kf_var": round(kf_var, 3),
        "skf_var": round(skf_var, 3),
        "overall_pos_ratio": round(overall_pos_ratio, 2),
        "non_nested_score": round(non_nested_score, 2),
        "nested_mean": round(nested_mean, 2),
        "nested_std": round(nested_std, 2),
        "selection_bias": round(selection_bias, 2),
        "outer_scores": [round(s, 2) for s in nested_scores],
        "chart": f"charts/{chart_path}",
    }


# ============================================================
# 3. CLASSIFICATION METRICS & CLASS-IMBALANCE IMPACT
# ============================================================
def _run_classification_metrics(df):
    target = "Class_Encoded"
    features = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    # Class Imbalance Simulation: The "Accuracy Paradox"
    # Artificially create a 95% Bad (0), 5% Good (1) severe imbalance dataset
    pos_idx = np.where(y.values == 1)[0]
    neg_idx = np.where(y.values == 0)[0]
    np.random.seed(42)
    n_pos_kept = int(len(neg_idx) * (0.05 / 0.95))
    sampled_pos = np.random.choice(pos_idx, size=min(n_pos_kept, len(pos_idx)), replace=False)
    imb_idx = np.concatenate([neg_idx, sampled_pos])
    X_imb = X.iloc[imb_idx]
    y_imb = y.iloc[imb_idx]

    X_imb_train, X_imb_test, y_imb_train, y_imb_test = train_test_split(
        X_imb, y_imb, test_size=0.25, random_state=42, stratify=y_imb
    )

    # Naive baseline: Always predict majority class 0
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_imb_train, y_imb_train)
    dummy_pred = dummy.predict(X_imb_test)
    dummy_acc = accuracy_score(y_imb_test, dummy_pred)
    dummy_rec = recall_score(y_imb_test, dummy_pred, zero_division=0)
    dummy_f1 = f1_score(y_imb_test, dummy_pred, zero_division=0)

    # Trained model on imbalanced data
    clf_imb = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42)
    clf_imb.fit(X_imb_train, y_imb_train)
    imb_prob = clf_imb.predict_proba(X_imb_test)[:, 1]
    imb_pred = clf_imb.predict(X_imb_test)
    imb_acc = accuracy_score(y_imb_test, imb_pred)
    imb_prec = precision_score(y_imb_test, imb_pred, zero_division=0)
    imb_rec = recall_score(y_imb_test, imb_pred, zero_division=0)
    imb_f1 = f1_score(y_imb_test, imb_pred, zero_division=0)
    imb_roc_auc = roc_auc_score(y_imb_test, imb_prob)
    imb_pr_auc = average_precision_score(y_imb_test, imb_prob)

    # Curves calculation
    fpr_bal, tpr_bal, _ = roc_curve(y_test, y_prob)
    prec_bal, rec_bal, _ = precision_recall_curve(y_test, y_prob)

    fpr_imb, tpr_imb, _ = roc_curve(y_imb_test, imb_prob)
    prec_imb, rec_imb, _ = precision_recall_curve(y_imb_test, imb_prob)

    # Chart Generation
    fig, axes = _setup_figure(figsize=(11, 4.6))

    # Left: ROC Curves (Balanced vs Imbalanced)
    axes[0].plot(fpr_bal, tpr_bal, color=PURPLE, linewidth=2.2, label=f"Balanced (AUC = {roc_auc:.3f})")
    axes[0].plot(fpr_imb, tpr_imb, color=PINK, linewidth=2.2, label=f"Imbalanced (5% pos, AUC = {imb_roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.2, label="Random Guess (AUC=0.50)")
    axes[0].set_title("ROC-AUC — Invariant to Class Prevalence", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(fontsize=8.5, loc="lower right")

    # Right: PR Curves (Balanced vs Imbalanced)
    axes[1].plot(rec_bal, prec_bal, color=PURPLE, linewidth=2.2, label=f"Balanced PR-AUC = {pr_auc:.3f}")
    axes[1].plot(rec_imb, prec_imb, color=PINK, linewidth=2.2, label=f"Imbalanced PR-AUC = {imb_pr_auc:.3f}")
    imb_baseline = float(np.mean(y_imb_test))
    axes[1].axhline(imb_baseline, color=AMBER, linestyle=":", linewidth=1.5, label=f"Imbalanced Baseline ({imb_baseline:.2f})")
    axes[1].set_title("PR-AUC — Exposing False Alarms in Imbalance", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Recall (TPR)")
    axes[1].set_ylabel("Precision (PPV)")
    axes[1].legend(fontsize=8.5, loc="upper right")

    fig.tight_layout()
    chart_path = "eval_classification_metrics.png"
    fig.savefig(CHART_DIR / chart_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(roc_auc, 3),
        "pr_auc": round(pr_auc, 3),
        "cm": cm.tolist(),
        "dummy_acc": round(dummy_acc * 100, 2),
        "dummy_rec": round(dummy_rec * 100, 2),
        "dummy_f1": round(dummy_f1 * 100, 2),
        "imb_acc": round(imb_acc * 100, 2),
        "imb_prec": round(imb_prec * 100, 2),
        "imb_rec": round(imb_rec * 100, 2),
        "imb_f1": round(imb_f1 * 100, 2),
        "imb_roc_auc": round(imb_roc_auc, 3),
        "imb_pr_auc": round(imb_pr_auc, 3),
        "chart": f"charts/{chart_path}",
    }


# ============================================================
# 4. REGRESSION METRICS & OUTLIER SENSITIVITY
# ============================================================
def _run_regression_metrics(df):
    target = "CO2 (pmm)"
    features = ["Temp", "Humid (%)", "Light (Fux)"]
    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Clean metrics
    clean_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    clean_mae = float(mean_absolute_error(y_test, y_pred))
    clean_mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100)
    clean_r2 = float(r2_score(y_test, y_pred))

    # Outlier Sensitivity Experiment: Inject sensor glitches (+300 to +800 CO2 spikes)
    contamination_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    rmse_curve = []
    mae_curve = []
    mape_curve = []

    np.random.seed(42)
    y_test_arr = y_test.values.copy()
    n_test = len(y_test_arr)

    for p in contamination_levels:
        y_corrupted = y_test_arr.copy()
        n_outliers = int(n_test * p)
        if n_outliers > 0:
            outlier_indices = np.random.choice(n_test, size=n_outliers, replace=False)
            y_corrupted[outlier_indices] += np.random.uniform(300, 800, size=n_outliers)

        rmse_curve.append(float(np.sqrt(mean_squared_error(y_corrupted, y_pred))))
        mae_curve.append(float(mean_absolute_error(y_corrupted, y_pred)))
        mape_curve.append(float(mean_absolute_percentage_error(y_corrupted, y_pred) * 100))

    # At 2% contamination:
    idx_2pct = 3
    outlier_rmse = rmse_curve[idx_2pct]
    outlier_mae = mae_curve[idx_2pct]
    rmse_surge = ((outlier_rmse - clean_rmse) / clean_rmse) * 100
    mae_surge = ((outlier_mae - clean_mae) / clean_mae) * 100

    # Chart Generation
    fig, axes = _setup_figure(figsize=(11, 4.6))

    # Left: Outlier Contamination vs Error Growth
    pct_labels = [f"{p*100:.1f}%" for p in contamination_levels]
    axes[0].plot(contamination_levels, rmse_curve, marker="o", color=PINK, linewidth=2.4, label="RMSE (L2 Squared Penalty)")
    axes[0].plot(contamination_levels, mae_curve, marker="s", color=PURPLE, linewidth=2.4, label="MAE (L1 Absolute Penalty)")
    axes[0].scatter([0.02], [outlier_rmse], color=PINK, s=90, zorder=5)
    axes[0].scatter([0.02], [outlier_mae], color=PURPLE, s=90, zorder=5)
    axes[0].set_title("Metric Sensitivity to Sensor Spikes (L1 vs L2)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Outlier Contamination Rate (%)")
    axes[0].set_ylabel("Error Value (pmm)")
    axes[0].legend(fontsize=8.5, loc="upper left")

    # Right: Error Residual Loss Shape (L1 vs L2 visualization)
    residuals = np.linspace(-40, 40, 200)
    l1_loss = np.abs(residuals)
    l2_loss = 0.05 * (residuals ** 2)
    axes[1].plot(residuals, l1_loss, color=PURPLE, linewidth=2.2, label="MAE Penalty: |e| (Linear)")
    axes[1].plot(residuals, l2_loss, color=PINK, linewidth=2.2, linestyle="--", label="RMSE Penalty: e² (Quadratic)")
    axes[1].set_title("Loss Penalty Comparison — Why RMSE Explodes on Outliers", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Error Residual (y - ŷ)")
    axes[1].set_ylabel("Penalty / Weight Assigned")
    axes[1].legend(fontsize=8.5, loc="upper center")

    fig.tight_layout()
    chart_path = "eval_regression_metrics.png"
    fig.savefig(CHART_DIR / chart_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "clean_rmse": round(clean_rmse, 3),
        "clean_mae": round(clean_mae, 3),
        "clean_mape": round(clean_mape, 2),
        "clean_r2": round(clean_r2, 4),
        "outlier_rmse": round(outlier_rmse, 3),
        "outlier_mae": round(outlier_mae, 3),
        "rmse_surge": round(rmse_surge, 1),
        "mae_surge": round(mae_surge, 1),
        "chart": f"charts/{chart_path}",
    }


# ============================================================
# 5. PROBABILITY CALIBRATION
# ============================================================
def _run_probability_calibration(df):
    target = "Class_Encoded"
    features = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Base uncalibrated model (RandomForest tends to predict compressed probabilities)
    base_rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    base_rf.fit(X_train, y_train)

    # 1. Platt Scaling (Logistic Sigmoid)
    platt_cal = CalibratedClassifierCV(base_rf, method="sigmoid", cv=3)
    platt_cal.fit(X_train, y_train)

    # 2. Isotonic Regression (Non-parametric step function)
    iso_cal = CalibratedClassifierCV(base_rf, method="isotonic", cv=3)
    iso_cal.fit(X_train, y_train)

    # Predictions
    p_uncal = base_rf.predict_proba(X_test)[:, 1]
    p_platt = platt_cal.predict_proba(X_test)[:, 1]
    p_iso = iso_cal.predict_proba(X_test)[:, 1]

    # Metrics
    brier_uncal = brier_score_loss(y_test, p_uncal)
    brier_platt = brier_score_loss(y_test, p_platt)
    brier_iso = brier_score_loss(y_test, p_iso)

    logloss_uncal = log_loss(y_test, p_uncal)
    logloss_platt = log_loss(y_test, p_platt)
    logloss_iso = log_loss(y_test, p_iso)

    # Calibration curves (Reliability diagrams)
    fop_uncal, mp_uncal = calibration_curve(y_test, p_uncal, n_bins=10)
    fop_platt, mp_platt = calibration_curve(y_test, p_platt, n_bins=10)
    fop_iso, mp_iso = calibration_curve(y_test, p_iso, n_bins=10)

    # Expected Calibration Error (ECE) approximation
    ece_uncal = float(np.mean(np.abs(fop_uncal - mp_uncal)))
    ece_platt = float(np.mean(np.abs(fop_platt - mp_platt)))
    ece_iso = float(np.mean(np.abs(fop_iso - mp_iso)))

    # Chart Generation
    fig, axes = _setup_figure(figsize=(11, 4.6))

    # Left: Reliability Diagram (Calibration Curves)
    axes[0].plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.5, label="Perfectly Calibrated (y = x)")
    axes[0].plot(mp_uncal, fop_uncal, marker="o", color=PINK, linewidth=2.0, label=f"Uncalibrated RF (Brier={brier_uncal:.4f})")
    axes[0].plot(mp_platt, fop_platt, marker="s", color=AMBER, linewidth=2.0, label=f"Platt Scaling / Sigmoid (Brier={brier_platt:.4f})")
    axes[0].plot(mp_iso, fop_iso, marker="^", color=GREEN, linewidth=2.0, label=f"Isotonic Regression (Brier={brier_iso:.4f})")
    axes[0].set_title("Reliability Diagram — Predicted Prob. vs True Frequency", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives (Empirical)")
    axes[0].legend(fontsize=8, loc="upper left")

    # Right: Probability Density Distributions
    axes[1].hist(p_uncal, bins=20, alpha=0.45, color=PINK, label="Uncalibrated (Clustered away from extremes)", density=True)
    axes[1].hist(p_iso, bins=20, alpha=0.45, color=GREEN, label="Isotonic (Smooth Posterior Spread)", density=True)
    axes[1].set_title("Predicted Probability Distribution Spread", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Predicted Probability P(Y=1)")
    axes[1].set_ylabel("Sample Density")
    axes[1].legend(fontsize=8, loc="upper center")

    fig.tight_layout()
    chart_path = "eval_probability_calibration.png"
    fig.savefig(CHART_DIR / chart_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "brier_uncal": round(brier_uncal, 4),
        "brier_platt": round(brier_platt, 4),
        "brier_iso": round(brier_iso, 4),
        "logloss_uncal": round(logloss_uncal, 4),
        "logloss_platt": round(logloss_platt, 4),
        "logloss_iso": round(logloss_iso, 4),
        "ece_uncal": round(ece_uncal * 100, 2),
        "ece_platt": round(ece_platt * 100, 2),
        "ece_iso": round(ece_iso * 100, 2),
        "chart": f"charts/{chart_path}",
    }


# ============================================================
# 6. HYPERPARAMETER SEARCH: GRID vs RANDOM vs BAYESIAN GP
# ============================================================
def _run_hyperparameter_search(df):
    target = "Class_Encoded"
    features = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
    X = df[features].values
    y = df[target].values

    # Small fast search budget for demonstration
    base_clf = RandomForestClassifier(random_state=42)

    # 1. Grid Search (16 evaluations)
    grid_params = {
        "n_estimators": [20, 50, 100, 150],
        "max_depth": [3, 6, 10, 15]
    }
    gs = GridSearchCV(base_clf, grid_params, cv=3, scoring="accuracy")
    gs.fit(X[:1200], y[:1200])
    grid_best = float(gs.best_score_ * 100)

    # 2. Random Search (16 evaluations - Bergstra & Bengio)
    dist_params = {
        "n_estimators": stats.randint(20, 200),
        "max_depth": stats.randint(3, 18)
    }
    rs = RandomizedSearchCV(base_clf, dist_params, n_iter=16, cv=3, scoring="accuracy", random_state=42)
    rs.fit(X[:1200], y[:1200])
    rand_best = float(rs.best_score_ * 100)

    # 3. Bayesian Optimisation with Gaussian Process (1D preview level demonstration)
    # Search over max_depth from 1 to 20
    candidate_depths = np.linspace(1, 20, 200).reshape(-1, 1)

    # Simulated true objective function with realistic noise and optima around depth=8
    def blackbox_objective(depths_arr):
        return 0.88 - 0.0006 * (depths_arr.ravel() - 8) ** 2 + 0.005 * np.sin(depths_arr.ravel())

    # Initial 5 evaluated points
    observed_x = np.array([2, 5, 11, 14, 18]).reshape(-1, 1)
    observed_y = blackbox_objective(observed_x) + np.random.normal(0, 0.002, size=len(observed_x))

    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-4, random_state=42)
    gp.fit(observed_x, observed_y)

    mu, sigma = gp.predict(candidate_depths, return_std=True)

    # Acquisition function: Upper Confidence Bound (UCB = mu + kappa * sigma)
    kappa = 2.0
    ucb = mu + kappa * sigma
    next_query_idx = int(np.argmax(ucb))
    next_query_x = float(candidate_depths[next_query_idx, 0])
    next_query_ucb = float(ucb[next_query_idx])

    # Chart Generation
    fig, axes = _setup_figure(figsize=(11, 4.6))

    # Left: Bergstra-Bengio Grid vs Random Search Concept in 2D Space
    np.random.seed(42)
    grid_x, grid_y = np.meshgrid([20, 50, 100, 150], [3, 6, 10, 15])
    axes[0].scatter(grid_x.ravel(), grid_y.ravel(), color=AMBER, s=40, label="Grid Points (Only 4 distinct depths!)", marker="s")
    axes[0].scatter(
        np.random.uniform(20, 150, 16),
        np.random.uniform(3, 15, 16),
        color=PURPLE,
        s=55,
        label="Random Points (16 distinct depths!)",
        marker="o",
    )
    axes[0].set_title("Bergstra-Bengio: Low Effective Dimensionality", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Hyperparameter 1: n_estimators")
    axes[0].set_ylabel("Hyperparameter 2: max_depth (Dominant)")
    axes[0].legend(fontsize=8, loc="upper right")

    # Right: Bayesian GP Surrogate & Acquisition Function Preview
    axes[1].plot(candidate_depths, mu * 100, color=PURPLE, linewidth=2.0, label="GP Surrogate Mean μ(x)")
    axes[1].fill_between(
        candidate_depths.ravel(),
        (mu - 2 * sigma) * 100,
        (mu + 2 * sigma) * 100,
        color=PURPLE,
        alpha=0.15,
        label="Epistemic Uncertainty ±2σ",
    )
    axes[1].scatter(observed_x, observed_y * 100, color=PINK, s=60, zorder=5, label="Sampled Points")
    axes[1].axvline(next_query_x, color=GREEN, linestyle="--", linewidth=1.8, label=f"Next Query Point (Depth={next_query_x:.1f})")
    axes[1].set_title("Bayesian Optimisation with Gaussian Process", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Hyperparameter (Tree Max Depth)")
    axes[1].set_ylabel("Expected Accuracy (%)")
    axes[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    chart_path = "eval_hyperparameter_search.png"
    fig.savefig(CHART_DIR / chart_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "grid_evals": 16,
        "grid_best": round(grid_best, 2),
        "rand_evals": 16,
        "rand_best": round(rand_best, 2),
        "next_query_x": round(next_query_x, 1),
        "gp_max_mu": round(float(np.max(mu) * 100), 2),
        "chart": f"charts/{chart_path}",
    }


# ============================================================
# 7. LEARNING CURVES & VALIDATION CURVES
# ============================================================
def _run_learning_curves(df):
    target = "Class_Encoded"
    features = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
    X = df[features].values
    y = df[target].values

    # 1. Learning Curve: performance vs training set size N
    train_sizes_fraction = [0.05, 0.15, 0.30, 0.50, 0.75, 1.0]
    clf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(
        clf, X, y, train_sizes=train_sizes_fraction, cv=4, scoring="accuracy", n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1) * 100
    train_std = np.std(train_scores, axis=1) * 100
    val_mean = np.mean(val_scores, axis=1) * 100
    val_std = np.std(val_scores, axis=1) * 100

    generalization_gap = float(train_mean[-1] - val_mean[-1])
    is_scarcity = bool(val_mean[-1] - val_mean[-2] > 0.8)

    # 2. Validation Curve: performance vs hyperparameter (max_depth)
    param_range = [1, 2, 4, 6, 8, 12, 16, 20]
    dt = DecisionTreeClassifier(random_state=42)
    vc_train_scores, vc_val_scores = validation_curve(
        dt, X, y, param_name="max_depth", param_range=param_range, cv=4, scoring="accuracy", n_jobs=-1
    )

    vc_train_mean = np.mean(vc_train_scores, axis=1) * 100
    vc_val_mean = np.mean(vc_val_scores, axis=1) * 100

    best_vc_idx = int(np.argmax(vc_val_mean))
    optimal_param = param_range[best_vc_idx]

    # Chart Generation
    fig, axes = _setup_figure(figsize=(11, 4.6))

    # Left: Learning Curve
    axes[0].plot(train_sizes, train_mean, marker="o", color=PURPLE, linewidth=2.0, label="Training Score")
    axes[0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color=PURPLE, alpha=0.12)
    axes[0].plot(train_sizes, val_mean, marker="s", color=GREEN, linewidth=2.0, label="Validation Score")
    axes[0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color=GREEN, alpha=0.12)
    axes[0].set_title(f"Learning Curve (Final Gap: {generalization_gap:.2f}%)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Training Set Size (N)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(fontsize=8.5, loc="lower right")

    # Right: Validation Curve (Diagnosing Under-fitting vs Over-fitting)
    axes[1].plot(param_range, vc_train_mean, marker="o", color=PINK, linewidth=2.0, label="Training Score")
    axes[1].plot(param_range, vc_val_mean, marker="s", color=BLUE, linewidth=2.0, label="Validation Score")
    axes[1].axvline(optimal_param, color=GREEN, linestyle="--", linewidth=1.5, label=f"Optimal Sweet Spot (Depth={optimal_param})")
    # Annotate Underfitting & Overfitting zones
    axes[1].axvspan(1, 3, color=AMBER, alpha=0.12, label="Under-fitting Zone (High Bias)")
    axes[1].axvspan(12, 20, color=PINK, alpha=0.10, label="Over-fitting Zone (High Variance)")
    axes[1].set_title("Validation Curve — Bias vs Variance Trade-off", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Model Complexity (max_depth)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=7.5, loc="lower right")

    fig.tight_layout()
    chart_path = "eval_learning_curves.png"
    fig.savefig(CHART_DIR / chart_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "final_train_score": round(train_mean[-1], 2),
        "final_val_score": round(val_mean[-1], 2),
        "generalization_gap": round(generalization_gap, 2),
        "is_scarcity": is_scarcity,
        "optimal_depth": optimal_param,
        "optimal_val_score": round(vc_val_mean[best_vc_idx], 2),
        "chart": f"charts/{chart_path}",
    }


# ============================================================
# 8. STATISTICAL SIGNIFICANCE TESTING
# ============================================================
def _run_significance_testing(df):
    target_cls = "Class_Encoded"
    features_cls = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
    X_cls = df[features_cls].values
    y_cls = df[target_cls].values

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.25, random_state=42, stratify=y_cls
    )

    # 1. McNemar's Test for Classifiers: Model A (Logistic Regression) vs Model B (Random Forest)
    model_a = LogisticRegression(max_iter=500, random_state=42)
    model_b = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42)

    model_a.fit(X_train_c, y_train_c)
    model_b.fit(X_train_c, y_train_c)

    pred_a = model_a.predict(X_test_c)
    pred_b = model_b.predict(X_test_c)

    correct_a = (pred_a == y_test_c)
    correct_b = (pred_b == y_test_c)

    # Contingency Table:
    # a: both correct
    # b: A correct, B incorrect
    # c: A incorrect, B correct
    # d: both incorrect
    n_a = int(np.sum(correct_a & correct_b))
    n_b = int(np.sum(correct_a & (~correct_b)))
    n_c = int(np.sum((~correct_a) & correct_b))
    n_d = int(np.sum((~correct_a) & (~correct_b)))

    # Edwards continuity-corrected McNemar Chi-Square
    discordant = n_b + n_c
    if discordant > 0:
        chi2_stat = float(((abs(n_b - n_c) - 1) ** 2) / discordant)
        p_val_mcnemar = float(1.0 - stats.chi2.cdf(chi2_stat, df=1))
    else:
        chi2_stat = 0.0
        p_val_mcnemar = 1.0

    acc_a = float(accuracy_score(y_test_c, pred_a) * 100)
    acc_b = float(accuracy_score(y_test_c, pred_b) * 100)
    mcnemar_significant = bool(p_val_mcnemar < 0.05)

    # 2. Paired Bootstrap for Regressors: Linear Regression vs Ridge/Tree Regressor
    target_reg = "CO2 (pmm)"
    features_reg = ["Temp", "Humid (%)", "Light (Fux)"]
    X_reg = df[features_reg].values
    y_reg = df[target_reg].values

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.25, random_state=42
    )

    reg_a = LinearRegression()
    reg_b = RandomForestRegressor(n_estimators=40, max_depth=6, random_state=42)

    reg_a.fit(X_train_r, y_train_r)
    reg_b.fit(X_train_r, y_train_r)

    pred_ra = reg_a.predict(X_test_r)
    pred_rb = reg_b.predict(X_test_r)

    # Paired Bootstrap: 1000 resamples of test set indices
    B = 1000
    n_test = len(y_test_r)
    np.random.seed(42)
    boot_diffs = np.zeros(B)

    for b_idx in range(B):
        idx = np.random.choice(n_test, size=n_test, replace=True)
        mae_a = np.mean(np.abs(y_test_r[idx] - pred_ra[idx]))
        mae_b = np.mean(np.abs(y_test_r[idx] - pred_rb[idx]))
        boot_diffs[b_idx] = mae_a - mae_b  # positive indicates Model B is better

    mean_diff = float(np.mean(boot_diffs))
    ci_lower, ci_upper = [float(v) for v in np.percentile(boot_diffs, [2.5, 97.5])]
    # One-sided bootstrap p-value: fraction where diff <= 0
    p_val_boot = float(np.mean(boot_diffs <= 0))
    boot_significant = bool(ci_lower > 0)

    # Chart Generation
    fig, axes = _setup_figure(figsize=(11, 4.6))

    # Left: McNemar Discordant Contingency Matrix
    cm_matrix = np.array([[n_a, n_b], [n_c, n_d]])
    im = axes[0].imshow(cm_matrix, cmap="Purples", alpha=0.85)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["RF Correct", "RF Incorrect"])
    axes[0].set_yticklabels(["LogReg Correct", "LogReg Incorrect"])
    for i in range(2):
        for j in range(2):
            val = cm_matrix[i, j]
            label = f"{val}\n" + ("(Agreement)" if i == j else ("(b: LogReg win)" if i == 0 else "(c: RF win)"))
            axes[0].text(j, i, label, ha="center", va="center", color="#33303e", fontweight="bold", fontsize=9.5)
    axes[0].set_title(f"McNemar's 2×2 Contingency Table (p={p_val_mcnemar:.4f})", fontsize=10.5, fontweight="bold")

    # Right: Paired Bootstrap Distribution of ΔMAE
    axes[1].hist(boot_diffs, bins=30, color=PURPLE, alpha=0.7, edgecolor="#ffffff")
    axes[1].axvline(0, color=PINK, linestyle="--", linewidth=1.8, label="Zero Improvement (H₀)")
    axes[1].axvline(mean_diff, color=GREEN, linewidth=2.2, label=f"Mean ΔMAE = {mean_diff:.2f}")
    axes[1].axvline(ci_lower, color=AMBER, linestyle=":", linewidth=1.8, label=f"95% CI [{ci_lower:.2f}, {ci_upper:.2f}]")
    axes[1].axvline(ci_upper, color=AMBER, linestyle=":", linewidth=1.8)
    axes[1].set_title("Paired Bootstrap: ΔMAE (Linear vs Random Forest)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Difference in MAE (Linear - Random Forest)")
    axes[1].set_ylabel("Bootstrap Replicates")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    chart_path = "eval_significance_testing.png"
    fig.savefig(CHART_DIR / chart_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "acc_a": round(acc_a, 2),
        "acc_b": round(acc_b, 2),
        "n_a": n_a,
        "n_b": n_b,
        "n_c": n_c,
        "n_d": n_d,
        "chi2_stat": round(chi2_stat, 3),
        "p_val_mcnemar": round(p_val_mcnemar, 5),
        "mcnemar_significant": mcnemar_significant,
        "mean_diff": round(mean_diff, 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "p_val_boot": round(p_val_boot, 5),
        "boot_significant": boot_significant,
        "chart": f"charts/{chart_path}",
    }


# ============================================================
# MASTER ENTRY POINT (CACHED)
# ============================================================
CACHE_FILE = BASE_DIR / "ml" / "evaluation_cache.json"

def get_model_evaluation(force_recompute=False):
    global _CACHE

    with _LOCK:
        if _CACHE is not None and not force_recompute:
            return _CACHE

        # 1. Check persistent disk cache (instantaneous < 0.002s load)
        if not force_recompute and CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                
                # Check that all charts exist in static
                all_charts_exist = all(
                    (BASE_DIR / "static" / v["chart"]).exists()
                    for v in cached_data.values()
                    if isinstance(v, dict) and "chart" in v
                )
                if all_charts_exist:
                    _CACHE = cached_data
                    return _CACHE
            except Exception:
                pass

        # 2. Recompute if cache is missing or explicitly requested
        df = load_preprocessed_data()

        split_res = _run_split_discipline(df)
        cv_res = _run_cross_validation(df)
        clf_res = _run_classification_metrics(df)
        reg_res = _run_regression_metrics(df)
        cal_res = _run_probability_calibration(df)
        search_res = _run_hyperparameter_search(df)
        curves_res = _run_learning_curves(df)
        sig_res = _run_significance_testing(df)

        _CACHE = {
            "split": split_res,
            "cv": cv_res,
            "classification": clf_res,
            "regression": reg_res,
            "calibration": cal_res,
            "search": search_res,
            "curves": curves_res,
            "significance": sig_res,
        }

        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_CACHE, f, indent=2)
        except Exception:
            pass

        return _CACHE
