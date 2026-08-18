
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from ml.data_loader import load_data


# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# FOLDER WHERE CHARTS ARE STORED
# ============================================================

CHART_DIR = BASE_DIR / "static" / "charts"

CHART_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COLOR THEME
# ============================================================

# Main green theme
GREEN = "#2E8B57"

# Dark green
DARK_GREEN = "#176B3A"

# Light green
LIGHT_GREEN = "#8FD19E"

# Blue
BLUE = "#4C78A8"

# Orange
ORANGE = "#F4A261"

# Red
RED = "#E76F51"

# Purple
PURPLE = "#8E6BBE"

# Background
BACKGROUND = "#F8FBF9"

# Grid
GRID_COLOR = "#DCE8E0"

# Text
TEXT_COLOR = "#173126"


# ============================================================
# SAVE FUNCTION
# ============================================================

def _save(fig, filename):
    """
    Save a matplotlib figure.
    """

    path = CHART_DIR / filename

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=140,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)


# ============================================================
# COMMON CHART STYLING
# ============================================================

def _style_axis(ax):
    """
    Apply common styling to all charts.
    """

    ax.set_facecolor(BACKGROUND)

    ax.title.set_color(TEXT_COLOR)

    ax.xaxis.label.set_color(TEXT_COLOR)

    ax.yaxis.label.set_color(TEXT_COLOR)

    ax.tick_params(
        colors=TEXT_COLOR,
        labelsize=9
    )

    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linestyle="--",
        linewidth=0.7,
        alpha=0.8
    )

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(GRID_COLOR)

    ax.spines["bottom"].set_color(GRID_COLOR)


# ============================================================
# EDA SUMMARY
# ============================================================

def get_eda_summary():

    df = load_data()


    # --------------------------------------------------------
    # Numerical columns in our dataset
    # --------------------------------------------------------

    numeric_cols = [
        "Temp",
        "Humid (%)",
        "Light (Fux)",
        "CO2 (pmm)"
    ]


    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_counts = (
        df["Class"]
        .value_counts()
        .to_dict()
    )


    # --------------------------------------------------------
    # Fruit distribution
    # --------------------------------------------------------

    fruit_counts = (
        df["Fruit"]
        .value_counts()
        .to_dict()
    )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = df.isna().sum()

    missing_percentage = (
        df.isna().mean() * 100
    ).round(2)


    # --------------------------------------------------------
    # Numerical descriptive statistics
    # --------------------------------------------------------

    numeric_description = (
        df[numeric_cols]
        .describe()
        .round(3)
        .reset_index()
        .to_dict(
            orient="records"
        )
    )


    # --------------------------------------------------------
    # Categorical statistics
    # --------------------------------------------------------

    categorical_description = (
        df[["Fruit", "Class"]]
        .describe()
        .reset_index()
        .fillna("")
        .to_dict(
            orient="records"
        )
    )


    # --------------------------------------------------------
    # Standardized class view
    # --------------------------------------------------------

    normalized_class = (
        df["Class"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "bad": "Bad",
            "good": "Good"
        })
    )


    normalized_counts = (
        normalized_class
        .value_counts()
        .to_dict()
    )


    # --------------------------------------------------------
    # Return EDA summary
    # --------------------------------------------------------

    return {

        "rows": int(
            df.shape[0]
        ),

        "columns": int(
            df.shape[1]
        ),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "columns_list": list(
            df.columns
        ),

        "missing": [

            {
                "column": column,

                "count": int(
                    missing[column]
                ),

                "percentage": float(
                    missing_percentage[column]
                )
            }

            for column in df.columns
        ],

        "numeric_description":
            numeric_description,

        "categorical_description":
            categorical_description,

        "class_counts":
            class_counts,

        "normalized_counts":
            normalized_counts,

        "fruit_counts":
            fruit_counts,

        "numeric_cols":
            numeric_cols
    }


# ============================================================
# GENERATE EDA CHARTS
# ============================================================

def generate_eda_charts():

    df = load_data()


    numeric_cols = [
        "Temp",
        "Humid (%)",
        "Light (Fux)",
        "CO2 (pmm)"
    ]


    # ========================================================
    # 1. CLASS DISTRIBUTION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    fig.patch.set_facecolor("white")

    counts = df["Class"].value_counts()


    # Give Good and Bad different colors
    colors = []

    for label in counts.index:

        label_text = str(label).strip().lower()

        if label_text == "good":

            colors.append(GREEN)

        elif label_text == "bad":

            colors.append(RED)

        else:

            colors.append(BLUE)


    ax.bar(
        counts.index.astype(str),
        counts.values,
        color=colors,
        edgecolor="white",
        linewidth=1.2
    )


    ax.set_title(
        "Fruit Quality Class Distribution",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR
    )

    ax.set_xlabel(
        "Class"
    )

    ax.set_ylabel(
        "Count"
    )


    _style_axis(ax)


    _save(
        fig,
        "class_distribution.png"
    )


    # ========================================================
    # 2. FRUIT DISTRIBUTION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    fig.patch.set_facecolor("white")

    counts = df["Fruit"].value_counts()


    fruit_colors = [
        GREEN,
        ORANGE,
        BLUE,
        PURPLE,
        LIGHT_GREEN,
        RED
    ]


    ax.bar(
        counts.index.astype(str),
        counts.values,
        color=fruit_colors[
            :len(counts)
        ],
        edgecolor="white",
        linewidth=1.2
    )


    ax.set_title(
        "Fruit Type Distribution",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR
    )

    ax.set_xlabel(
        "Fruit"
    )

    ax.set_ylabel(
        "Count"
    )


    plt.xticks(
        rotation=25,
        ha="right"
    )


    _style_axis(ax)


    _save(
        fig,
        "fruit_distribution.png"
    )


    # ========================================================
    # 3. TEMPERATURE DISTRIBUTION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    fig.patch.set_facecolor("white")


    ax.hist(
        df["Temp"].dropna(),
        bins=30,
        color=GREEN,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9
    )


    ax.set_title(
        "Temperature Distribution",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR
    )

    ax.set_xlabel(
        "Temperature"
    )

    ax.set_ylabel(
        "Frequency"
    )


    _style_axis(ax)


    _save(
        fig,
        "temperature_distribution.png"
    )


    # ========================================================
    # 4. HUMIDITY DISTRIBUTION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    fig.patch.set_facecolor("white")


    ax.hist(
        df["Humid (%)"].dropna(),
        bins=30,
        color=BLUE,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9
    )


    ax.set_title(
        "Humidity Distribution",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR
    )

    ax.set_xlabel(
        "Humidity (%)"
    )

    ax.set_ylabel(
        "Frequency"
    )


    _style_axis(ax)


    _save(
        fig,
        "humidity_distribution.png"
    )


    # ========================================================
    # 5. LIGHT DISTRIBUTION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    fig.patch.set_facecolor("white")


    ax.hist(
        df["Light (Fux)"].dropna(),
        bins=30,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9
    )


    ax.set_title(
        "Light Intensity Distribution",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR
    )

    ax.set_xlabel(
        "Light (Fux)"
    )

    ax.set_ylabel(
        "Frequency"
    )


    _style_axis(ax)


    _save(
        fig,
        "light_distribution.png"
    )


    # ========================================================
    # 6. CO2 DISTRIBUTION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    fig.patch.set_facecolor("white")


    ax.hist(
        df["CO2 (pmm)"].dropna(),
        bins=30,
        color=PURPLE,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9
    )


    ax.set_title(
        "CO₂ Level Distribution",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR
    )

    ax.set_xlabel(
        "CO₂ (pmm)"
    )

    ax.set_ylabel(
        "Frequency"
    )


    _style_axis(ax)


    _save(
        fig,
        "co2_distribution.png"
    )


    # ========================================================
    # 7. CORRELATION MATRIX
    # ========================================================

    correlation = df[
        numeric_cols
    ].corr()


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    fig.patch.set_facecolor("white")


    image = ax.imshow(
        correlation.values,
        interpolation="nearest",
        cmap="YlGn",
        vmin=-1,
        vmax=1
    )


    ax.set_xticks(
        range(len(numeric_cols))
    )

    ax.set_yticks(
        range(len(numeric_cols))
    )


    ax.set_xticklabels(
        numeric_cols,
        rotation=30,
        ha="right"
    )

    ax.set_yticklabels(
        numeric_cols
    )


    ax.set_title(
        "Correlation Matrix",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR
    )


    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Correlation",
        color=TEXT_COLOR
    )


    # --------------------------------------------------------
    # Display correlation values
    # --------------------------------------------------------

    for i in range(
        len(numeric_cols)
    ):

        for j in range(
            len(numeric_cols)
        ):

            value = correlation.iloc[
                i,
                j
            ]


            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=TEXT_COLOR
            )


    ax.set_facecolor(
        BACKGROUND
    )


    _save(
        fig,
        "correlation_heatmap.png"
    )


    # ========================================================
    # 8. BOXPLOTS BY FRUIT
    # ========================================================

    for column in numeric_cols:

        groups = []

        labels = []


        fruits = (
            df["Fruit"]
            .drop_duplicates()
        )


        for fruit in fruits:

            values = (
                df.loc[
                    df["Fruit"] == fruit,
                    column
                ]
                .dropna()
            )

            groups.append(
                values
            )

            labels.append(
                fruit
            )


        fig, ax = plt.subplots(
            figsize=(8, 4.8)
        )

        fig.patch.set_facecolor(
            "white"
        )


        box = ax.boxplot(
            groups,
            tick_labels=labels,
            patch_artist=True
        )


        # ----------------------------------------------------
        # Different colors for each fruit
        # ----------------------------------------------------

        for index, patch in enumerate(
            box["boxes"]
        ):

            patch.set_facecolor(
                fruit_colors[
                    index % len(fruit_colors)
                ]
            )

            patch.set_alpha(
                0.75
            )

            patch.set_edgecolor(
                DARK_GREEN
            )


        # Median line
        for median in box["medians"]:

            median.set_color(
                DARK_GREEN
            )

            median.set_linewidth(
                2
            )


        # Whiskers
        for whisker in box["whiskers"]:

            whisker.set_color(
                DARK_GREEN
            )


        # Caps
        for cap in box["caps"]:

            cap.set_color(
                DARK_GREEN
            )


        # Outliers
        for flier in box["fliers"]:

            flier.set_marker(
                "o"
            )

            flier.set_markersize(
                3
            )

            flier.set_alpha(
                0.45
            )


        ax.set_title(
            f"{column} by Fruit Type",
            fontsize=15,
            fontweight="bold",
            color=TEXT_COLOR
        )


        ax.set_xlabel(
            "Fruit"
        )

        ax.set_ylabel(
            column
        )


        plt.xticks(
            rotation=25,
            ha="right"
        )


        _style_axis(ax)


        # ----------------------------------------------------
        # Generate the same filenames as before
        # ----------------------------------------------------

        filename = (
            column
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "pct")
        )


        _save(
            fig,
            f"{filename}_by_fruit.png"
        )


# ============================================================
# END OF EDA
# ============================================================

