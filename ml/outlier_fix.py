import pandas as pd

from ml.data_loader import load_data


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()

print("Original dataset shape:")
print(df.shape)


# ============================================================
# NUMERICAL COLUMNS
# ============================================================

numeric_cols = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]


# ============================================================
# PROCESS EACH COLUMN
# ============================================================

outlier_summary = []


for feature in numeric_cols:

    print("\n" + "=" * 60)

    print(
        "Processing:",
        feature
    )

    print("=" * 60)


    # --------------------------------------------------------
    # ORIGINAL STATISTICS
    # --------------------------------------------------------

    print("\nOriginal Statistics:")

    print(
        df[feature].describe()
    )


    # --------------------------------------------------------
    # Q1 AND Q3
    # --------------------------------------------------------

    Q1 = df[feature].quantile(
        0.25
    )

    Q3 = df[feature].quantile(
        0.75
    )


    # --------------------------------------------------------
    # IQR
    # --------------------------------------------------------

    IQR = Q3 - Q1


    # --------------------------------------------------------
    # FENCES
    # --------------------------------------------------------

    lower_fence = (
        Q1 - 1.5 * IQR
    )

    upper_fence = (
        Q3 + 1.5 * IQR
    )


    print("\nQ1 =", Q1)

    print("Q3 =", Q3)

    print("IQR =", IQR)

    print(
        "Lower fence =",
        lower_fence
    )

    print(
        "Upper fence =",
        upper_fence
    )


    # --------------------------------------------------------
    # FIND OUTLIERS
    # --------------------------------------------------------

    outliers = df[
        (df[feature] < lower_fence)
        |
        (df[feature] > upper_fence)
    ]


    print(
        "\nNumber of outliers =",
        len(outliers)
    )


    # --------------------------------------------------------
    # CLIP OUTLIERS
    # --------------------------------------------------------

    clipped_column = (
        feature + "_Clipped"
    )


    df[clipped_column] = (
        df[feature]
        .clip(
            lower=lower_fence,
            upper=upper_fence
        )
    )


    # --------------------------------------------------------
    # BEFORE / AFTER
    # --------------------------------------------------------

    print(
        "\nMinimum BEFORE clipping:"
    )

    print(
        df[feature].min()
    )


    print(
        "\nMinimum AFTER clipping:"
    )

    print(
        df[clipped_column].min()
    )


    print(
        "\nMaximum BEFORE clipping:"
    )

    print(
        df[feature].max()
    )


    print(
        "\nMaximum AFTER clipping:"
    )

    print(
        df[clipped_column].max()
    )


    outlier_summary.append({

        "Feature": feature,

        "Q1": round(Q1, 3),

        "Q3": round(Q3, 3),

        "IQR": round(IQR, 3),

        "Lower_Fence":
            round(lower_fence, 3),

        "Upper_Fence":
            round(upper_fence, 3),

        "Outlier_Count":
            len(outliers)

    })


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    outlier_summary
)


print("\n\nOUTLIER SUMMARY")

print(
    summary_df
)