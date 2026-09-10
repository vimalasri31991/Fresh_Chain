from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHART_DIR = BASE_DIR / "static" / "charts"

CHART_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COLUMNS
# ============================================================

NUMERIC_COLUMNS = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]

CATEGORICAL_COLUMNS = [
    "Fruit",
    "Class"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_eda_data():

    from ml.data_loader import load_data

    df = load_data().copy()

    return df


# ============================================================
# NUMERIC SUMMARY
# ============================================================

def get_numeric_summary(df):

    numeric_df = df[
        NUMERIC_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    summary = []

    for column in NUMERIC_COLUMNS:

        series = numeric_df[column].dropna()

        if series.empty:

            summary.append({
                "column": column,
                "count": 0,
                "mean": 0,
                "std": 0,
                "min": 0,
                "q25": 0,
                "median": 0,
                "q75": 0,
                "max": 0
            })

            continue

        summary.append({

            "column": column,

            "count": int(
                series.count()
            ),

            "mean": round(
                float(series.mean()),
                3
            ),

            "std": round(
                float(series.std()),
                3
            ),

            "min": round(
                float(series.min()),
                3
            ),

            "q25": round(
                float(series.quantile(0.25)),
                3
            ),

            "median": round(
                float(series.median()),
                3
            ),

            "q75": round(
                float(series.quantile(0.75)),
                3
            ),

            "max": round(
                float(series.max()),
                3
            )
        })

    return summary


# ============================================================
# GENERATE EDA CHARTS
# ============================================================

def generate_eda_charts():

    df = load_eda_data()

    # --------------------------------------------------------
    # Make sure numeric columns are numeric
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


    # ========================================================
    # 1. TEMPERATURE DISTRIBUTION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    values = df["Temp"].dropna()

    ax.hist(
        values,
        bins=20,
        edgecolor="white",
        alpha=0.85,
        color="#ff7043"
    )

    ax.set_title(
        "Temperature Distribution",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Temperature"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        CHART_DIR / "eda_temperature.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ========================================================
    # 2. HUMIDITY DISTRIBUTION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    values = df["Humid (%)"].dropna()

    ax.hist(
        values,
        bins=20,
        edgecolor="white",
        alpha=0.85,
        color="#42a5f5"
    )

    ax.set_title(
        "Humidity Distribution",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Humidity (%)"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        CHART_DIR / "eda_humidity.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ========================================================
    # 3. LIGHT DISTRIBUTION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    values = df["Light (Fux)"].dropna()

    ax.hist(
        values,
        bins=20,
        edgecolor="white",
        alpha=0.85,
        color="#ab47bc"
    )

    ax.set_title(
        "Light Distribution",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Light (Fux)"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        CHART_DIR / "eda_light.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ========================================================
    # 4. CO2 DISTRIBUTION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    values = df["CO2 (pmm)"].dropna()

    ax.hist(
        values,
        bins=20,
        edgecolor="white",
        alpha=0.85,
        color="#26a69a"
    )

    ax.set_title(
        "CO₂ Distribution",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "CO₂ (pmm)"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        CHART_DIR / "eda_co2.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ========================================================
    # 5. FRUIT DISTRIBUTION
    # ========================================================

    fruit_counts = (
        df["Fruit"]
        .dropna()
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    colors = [
        "#ff9800",
        "#66bb6a",
        "#42a5f5",
        "#ef5350"
    ]

    bars = ax.bar(
        fruit_counts.index.astype(str),
        fruit_counts.values,
        color=colors[:len(fruit_counts)]
    )

    ax.set_title(
        "Fruit Type Distribution",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Fruit"
    )

    ax.set_ylabel(
        "Number of Records"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    fig.tight_layout()

    fig.savefig(
        CHART_DIR / "eda_fruit_distribution.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ========================================================
    # 6. CLASS DISTRIBUTION
    # ========================================================

    class_counts = (
        df["Class"]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    class_colors = []

    for class_name in class_counts.index:

        if class_name == "Good":

            class_colors.append("#2ecc71")

        elif class_name == "BAD":

            class_colors.append("#e74c3c")

        else:

            class_colors.append("#3498db")

    bars = ax.bar(
        class_counts.index,
        class_counts.values,
        color=class_colors
    )

    ax.set_title(
        "Fruit Quality Class Distribution",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Class"
    )

    ax.set_ylabel(
        "Number of Records"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    fig.tight_layout()

    fig.savefig(
        CHART_DIR / "eda_class_distribution.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ========================================================
    # 7. NUMERIC BOX PLOTS
    # ========================================================

    numeric_data = []

    valid_labels = []

    for column in NUMERIC_COLUMNS:

        values = df[column].dropna()

        if len(values) > 0:

            numeric_data.append(
                values.values
            )

            valid_labels.append(
                column
            )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    if numeric_data:

        box = ax.boxplot(
            numeric_data,
            patch_artist=True
        )

        box_colors = [
            "#ff7043",
            "#42a5f5",
            "#ab47bc",
            "#26a69a"
        ]

        for patch, color in zip(
            box["boxes"],
            box_colors
        ):

            patch.set_facecolor(
                color
            )

            patch.set_alpha(
                0.75
            )

        ax.set_xticks(
            range(
                1,
                len(valid_labels) + 1
            )
        )

        ax.set_xticklabels(
            valid_labels
        )

    ax.set_title(
        "Environmental Variables - Box Plot",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Value"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        CHART_DIR / "eda_boxplot.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ========================================================
    # 8. CORRELATION HEATMAP
    # ========================================================

    correlation = df[
        NUMERIC_COLUMNS
    ].corr()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    image = ax.imshow(
        correlation,
        cmap="RdYlBu",
        vmin=-1,
        vmax=1
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation"
    )

    ax.set_xticks(
        range(len(NUMERIC_COLUMNS))
    )

    ax.set_yticks(
        range(len(NUMERIC_COLUMNS))
    )

    ax.set_xticklabels(
        NUMERIC_COLUMNS,
        rotation=35,
        ha="right"
    )

    ax.set_yticklabels(
        NUMERIC_COLUMNS
    )

    ax.set_title(
        "Correlation Between Environmental Variables",
        fontsize=15,
        fontweight="bold"
    )

    for i in range(
        len(NUMERIC_COLUMNS)
    ):

        for j in range(
            len(NUMERIC_COLUMNS)
        ):

            value = correlation.iloc[i, j]

            if pd.notna(value):

                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontweight="bold"
                )

    fig.tight_layout()

    fig.savefig(
        CHART_DIR / "eda_correlation.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ========================================================
    # 9. TEMPERATURE VS CO2
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.scatter(
        df["Temp"],
        df["CO2 (pmm)"],
        alpha=0.65,
        color="#8e44ad",
        edgecolors="white"
    )

    ax.set_title(
        "Temperature vs CO₂",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Temperature"
    )

    ax.set_ylabel(
        "CO₂ (pmm)"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        CHART_DIR / "eda_temp_vs_co2.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# EDA SUMMARY
# ============================================================

def get_eda_summary():

    df = load_eda_data()


    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    rows = int(
        df.shape[0]
    )

    columns_count = int(
        df.shape[1]
    )

    columns = [
        str(column)
        for column in df.columns
    ]


    # ========================================================
    # CLASS INFORMATION
    # ========================================================

    class_series = (
        df["Class"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    classes = sorted(
        class_series.unique().tolist()
    )


    # ========================================================
    # FRUIT INFORMATION
    # ========================================================

    fruit_series = (
        df["Fruit"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    fruit_types = sorted(
        fruit_series.unique().tolist()
    )


    # ========================================================
    # MISSING VALUES
    # ========================================================

    missing_values = []

    for column in df.columns:

        missing_values.append({

            "column": str(column),

            "missing": int(
                df[column].isna().sum()
            )

        })


    # ========================================================
    # DATA TYPES
    # ========================================================

    data_types = []

    for column in df.columns:

        data_types.append({

            "column": str(column),

            "dtype": str(
                df[column].dtype
            )

        })


    # ========================================================
    # NUMERIC SUMMARY
    # ========================================================

    numeric_summary = get_numeric_summary(
        df
    )


    # ========================================================
    # CLASS COUNTS
    # ========================================================

    class_counts = []

    for class_name, count in (
        class_series.value_counts().items()
    ):

        class_counts.append({

            "class": str(class_name),

            "count": int(count)

        })


    # ========================================================
    # FRUIT COUNTS
    # ========================================================

    fruit_counts = []

    for fruit_name, count in (
        fruit_series.value_counts().items()
    ):

        fruit_counts.append({

            "fruit": str(fruit_name),

            "count": int(count)

        })


    # ========================================================
    # RETURN SUMMARY
    # ========================================================

    return {

        "n_rows": rows,

        "n_cols": columns_count,

        "columns": columns,

        "classes": classes,

        "n_classes": len(classes),

        "fruit_types": fruit_types,

        "n_fruit_types": len(fruit_types),

        "numeric_columns": NUMERIC_COLUMNS,

        "numeric_summary": numeric_summary,

        "missing_values": missing_values,

        "data_types": data_types,

        "class_counts": class_counts,

        "fruit_counts": fruit_counts
    }