"""
Exploratory Data Analysis - 15 tasks for the FreshChain dataset.

Tasks (same structure as the reference EDA script, adapted to the
fruit dataset: Fruit, Temp, Humid (%), Light (Fux), CO2 (pmm), Class)

 1 Load data              9 Relationship plots
 2 Basic info            10 Categorical feature counts
 3 Missing values        11 Fruit vs target class
 4 Duplicate rows        12 Conditions by fruit
 5 Target distribution   13 Average readings by fruit
 6 Numeric distributions 14 Class-wise analysis
 7 Outlier detection     15 Pairplot
 8 Correlation heatmap
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml.data_loader import load_data, DATA_PATH


BASE_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = BASE_DIR / "static" / "charts"

NUMERIC = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
TARGET = "Class"
CATEGORY = "Fruit"

PURPLE = "#7c58c9"
PINK = "#c1477e"
GREEN = "#4c9a6c"
AMBER = "#c68a3d"

CLASS_COLORS = {"BAD": PINK, "Good": GREEN}
FRUIT_COLORS = [PURPLE, PINK, GREEN, AMBER]


# ============================================================
# HELPERS
# ============================================================

def _data():

    df = load_data().copy()

    df[TARGET] = (
        df[TARGET].astype(str).str.strip().str.lower()
        .map({"bad": "BAD", "good": "Good"})
    )

    return df


def _save(fig, name):

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(CHART_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return f"charts/{name}"


def _order(df):
    return ["BAD", "Good"] if "BAD" in set(df[TARGET]) else sorted(df[TARGET].unique())


# ============================================================
# CHARTS  (one function per task that needs a picture)
# ============================================================

def _chart_task5(df):

    order = _order(df)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    sns.countplot(data=df, x=TARGET, order=order, hue=TARGET,
                  palette=CLASS_COLORS, legend=False, ax=ax)

    for p in ax.patches:
        ax.annotate(int(p.get_height()),
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Target class")
    ax.set_ylabel("Count")
    ax.set_title("Target Class Distribution")

    return _save(fig, "eda_task_05.png")


def _chart_task6(df):

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    for ax, col, color in zip(axes.ravel(), NUMERIC, FRUIT_COLORS):
        ax.hist(df[col], bins=20, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(df[col].mean(), color="#33303e", linestyle="--",
                   label=f"Mean {df[col].mean():.1f}")
        ax.set_title(col)
        ax.legend(fontsize=8)

    fig.suptitle("Numeric Feature Distributions (with mean line)")

    return _save(fig, "eda_task_06.png")


def _chart_task7(df):

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))

    for ax, col, color in zip(axes.ravel(), NUMERIC, FRUIT_COLORS):
        sns.boxplot(x=df[col], color=color, ax=ax, fliersize=2)
        ax.set_title(f"Boxplot - {col}")

    return _save(fig, "eda_task_07.png")


def _chart_task8(df):

    corr = df[NUMERIC].corr().round(2)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Heatmap")

    return _save(fig, "eda_task_08.png")


def _chart_task9(df):

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (x, y) in zip(axes, [("Temp", "CO2 (pmm)"),
                                 ("Humid (%)", "Light (Fux)")]):
        sns.regplot(data=df.sample(min(2500, len(df)), random_state=42),
                    x=x, y=y, ax=ax,
                    scatter_kws={"alpha": 0.35, "s": 12, "color": PURPLE},
                    line_kws={"color": PINK})
        ax.set_title(f"{x} vs {y}")

    return _save(fig, "eda_task_09.png")


def _chart_task10(df):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.countplot(data=df, x=CATEGORY, hue=CATEGORY,
                  palette=FRUIT_COLORS, legend=False, ax=axes[0])
    axes[0].set_title("Fruit Count")

    sns.countplot(data=df, x=TARGET, order=_order(df), hue=TARGET,
                  palette=CLASS_COLORS, legend=False, ax=axes[1])
    axes[1].set_title("Class Count")

    return _save(fig, "eda_task_10.png")


def _chart_task11(df):

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(data=df, x=CATEGORY, hue=TARGET,
                  hue_order=_order(df), palette=CLASS_COLORS, ax=ax)
    ax.set_title("Fruit vs Target Class")

    return _save(fig, "eda_task_11.png")


def _chart_task12(df):

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for ax, col in zip(axes.ravel(), NUMERIC):
        sns.boxplot(data=df, x=CATEGORY, y=col, hue=CATEGORY,
                    palette=FRUIT_COLORS, legend=False, ax=ax, fliersize=2)
        ax.set_title(f"{col} by Fruit")

    return _save(fig, "eda_task_12.png")


def _chart_task13(df):

    means = df.groupby(CATEGORY)[NUMERIC].mean()
    scaled = (means - means.min()) / (means.max() - means.min())

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for col, color in zip(NUMERIC, FRUIT_COLORS):
        ax.plot(scaled.index, scaled[col], marker="o", color=color, label=col)

    ax.set_ylabel("Average reading (0-1 scaled)")
    ax.set_title("Average Environmental Readings by Fruit")
    ax.legend(fontsize=8)

    return _save(fig, "eda_task_13.png")


def _chart_task14(df):

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for ax, col in zip(axes.ravel(), NUMERIC):
        sns.kdeplot(data=df, x=col, hue=TARGET, hue_order=_order(df),
                    palette=CLASS_COLORS, fill=True, common_norm=False,
                    warn_singular=False, ax=ax)
        ax.set_title(f"{col} by Class")

    return _save(fig, "eda_task_14.png")


def _chart_task15(df):

    sample = df[NUMERIC + [TARGET]].sample(min(1500, len(df)), random_state=42)

    grid = sns.pairplot(sample, hue=TARGET, hue_order=_order(df),
                        palette=CLASS_COLORS, diag_kind="hist",
                        plot_kws={"alpha": 0.5, "s": 14})

    grid.figure.suptitle("Pairwise Relationships", y=1.02)

    return _save(grid.figure, "eda_task_15.png")


CHART_BUILDERS = {
    5: _chart_task5, 6: _chart_task6, 7: _chart_task7, 8: _chart_task8,
    9: _chart_task9, 10: _chart_task10, 11: _chart_task11,
    12: _chart_task12, 13: _chart_task13, 14: _chart_task14,
    15: _chart_task15,
}


def _chart_path(task):
    return f"charts/eda_task_{task:02d}.png"


# ============================================================
# CHART GENERATION (cached: only runs when a file is missing
# or the dataset is newer than the charts)
# ============================================================

def generate_eda_charts(force=False):

    df = _data()

    data_time = DATA_PATH.stat().st_mtime

    with sns.axes_style("whitegrid"):

        for task, builder in CHART_BUILDERS.items():

            target = BASE_DIR / "static" / _chart_path(task)

            fresh = target.exists() and target.stat().st_mtime >= data_time

            if force or not fresh:
                builder(df)


# ============================================================
# SUMMARY  (everything the template shows)
# ============================================================

def _outlier_counts(df):

    rows = []

    for col in NUMERIC:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({
            "column": col,
            "outliers": count,
            "percent": round(count / len(df) * 100, 2),
        })

    return rows


def get_eda_summary():

    df = _data()
    n = len(df)

    class_counts = df[TARGET].value_counts()
    fruit_counts = df[CATEGORY].value_counts()

    classes = _order(df)

    class_rows = [
        {"class": c, "count": int(class_counts.get(c, 0)),
         "percent": round(class_counts.get(c, 0) / n * 100, 1)}
        for c in classes
    ]

    fruit_rows = [
        {"fruit": f, "count": int(v), "percent": round(v / n * 100, 1)}
        for f, v in fruit_counts.items()
    ]

    # ---- task 2 / 3 / 4 / 7 / 8 / 11 tables ----
    info_rows = [
        {"column": c, "dtype": str(df[c].dtype),
         "non_null": int(df[c].notna().sum()), "unique": int(df[c].nunique())}
        for c in df.columns
    ]

    describe = df[NUMERIC].describe().T.round(2).reset_index()
    describe = describe.rename(columns={"index": "feature"})

    missing = [
        {"column": c, "missing": int(df[c].isnull().sum()),
         "percent": round(df[c].isnull().mean() * 100, 2)}
        for c in df.columns
    ]

    duplicates = int(df.duplicated().sum())

    corr = df[NUMERIC].corr()
    pairs = []

    for i, a in enumerate(NUMERIC):
        for b in NUMERIC[i + 1:]:
            pairs.append({"pair": f"{a} ↔ {b}", "value": round(float(corr.loc[a, b]), 2)})

    pairs.sort(key=lambda p: abs(p["value"]), reverse=True)

    crosstab = pd.crosstab(df[CATEGORY], df[TARGET], normalize="index") * 100

    good_share = [
        {"fruit": f, "good": round(float(crosstab.loc[f].get("Good", 0)), 1),
         "bad": round(float(crosstab.loc[f].get("BAD", 0)), 1)}
        for f in crosstab.index
    ]

    group_means = df.groupby(TARGET)[NUMERIC].mean().round(2)

    class_means = [
        {"class": c, **{col: float(group_means.loc[c, col]) for col in NUMERIC}}
        for c in group_means.index
    ]

    tasks = [
        {"num": 1, "title": "Load Data", "badge": "Dataset",
         "chips": [("Rows", n), ("Columns", df.shape[1]),
                   ("File", DATA_PATH.name)],
         "table": {"headers": list(df.columns),
                   "rows": df.head(5).values.tolist()}},

        {"num": 2, "title": "Basic Info", "badge": "Structure",
         "chips": [("Numeric", len(NUMERIC)), ("Categorical", 2)],
         "table": {"headers": ["Column", "Type", "Non-null", "Unique"],
                   "rows": [[r["column"], r["dtype"], r["non_null"], r["unique"]]
                            for r in info_rows]},
         "table2": {"headers": list(describe.columns),
                    "rows": describe.values.tolist()}},

        {"num": 3, "title": "Missing Values", "badge": "Data Quality",
         "chips": [("Total missing", int(df.isnull().sum().sum())),
                   ("Columns checked", df.shape[1])],
         "bars": [{"label": r["column"], "value": r["missing"],
                   "percent": r["percent"], "text": f'{r["missing"]} missing'}
                  for r in missing]},

        {"num": 4, "title": "Duplicate Rows", "badge": "Data Quality",
         "chips": [("Duplicate rows", duplicates),
                   ("Share of data", f"{duplicates / n * 100:.1f}%"),
                   ("Unique rows", n - duplicates)],
         "bars": [{"label": "Duplicates", "value": duplicates,
                   "percent": round(duplicates / n * 100, 1),
                   "text": f"{duplicates / n * 100:.1f}%"},
                  {"label": "Unique rows", "value": n - duplicates,
                   "percent": round((n - duplicates) / n * 100, 1),
                   "text": f"{(n - duplicates) / n * 100:.1f}%"}]},

        {"num": 5, "title": "Target Class Distribution", "badge": "Target",
         "chips": [(r["class"], f'{r["count"]} ({r["percent"]}%)') for r in class_rows]},

        {"num": 6, "title": "Numeric Feature Distributions", "badge": "Univariate",
         "chips": [(c, f"mean {df[c].mean():.1f}") for c in NUMERIC]},

        {"num": 7, "title": "Outlier Detection (Boxplots)", "badge": "Outliers",
         "chips": [(r["column"], f'{r["outliers"]} outliers') for r in _outlier_counts(df)]},

        {"num": 8, "title": "Correlation Heatmap", "badge": "Multivariate",
         "chips": [(p["pair"], p["value"]) for p in pairs[:3]]},

        {"num": 9, "title": "Relationship Plots", "badge": "Bivariate",
         "chips": [("Temp ↔ CO2", round(float(corr.loc["Temp", "CO2 (pmm)"]), 2)),
                   ("Humid ↔ Light", round(float(corr.loc["Humid (%)", "Light (Fux)"]), 2))]},

        {"num": 10, "title": "Categorical Feature Counts", "badge": "Univariate",
         "chips": [(r["fruit"], r["count"]) for r in fruit_rows]},

        {"num": 11, "title": "Fruit vs Target Class", "badge": "Bivariate",
         "chips": [(r["fruit"], f'{r["good"]}% Good') for r in good_share]},

        {"num": 12, "title": "Conditions by Fruit", "badge": "Bivariate",
         "chips": [(f, f"{int(c)} rows") for f, c in fruit_counts.items()]},

        {"num": 13, "title": "Average Readings by Fruit", "badge": "Trend",
         "chips": [(c, f"{df[c].mean():.1f}") for c in NUMERIC]},

        {"num": 14, "title": "Class-wise Analysis", "badge": "Target vs Features",
         "chips": [(r["class"], f'Light {r["Light (Fux)"]} • CO₂ {r["CO2 (pmm)"]}')
                   for r in class_means]},

        {"num": 15, "title": "Pairplot", "badge": "Multivariate",
         "chips": [("Features", len(NUMERIC)), ("Hue", "Class"), ("Sample", min(1500, n))]},
    ]

    for t in tasks:
        if t["num"] in CHART_BUILDERS:
            t["image"] = _chart_path(t["num"])

    return {
        "n_rows": n,
        "n_cols": int(df.shape[1]),
        "n_classes": len(classes),
        "n_fruit_types": int(df[CATEGORY].nunique()),
        "classes": classes,
        "fruit_types": list(fruit_counts.index),
        "class_counts": class_rows,
        "fruit_counts": fruit_rows,
        "tasks": tasks,
    }