from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder
)

from ml.data_loader import load_data


# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PREPROCESSED_PATH = BASE_DIR / "preprocessed_data.csv"


# ============================================================
# PROJECT FEATURES
# ============================================================

NUMERIC_COLS = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]


CATEGORICAL_COLS = [
    "Fruit"
]


TARGET_COL = "Class"


CLASS_MAPPING = {
    "BAD": 0,
    "Good": 1
}


# ============================================================
# CLASS NORMALIZATION
# ============================================================

def normalize_class(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "bad": "BAD",
            "good": "Good"
        })
    )


# ============================================================
# ONE-HOT ENCODER
# ============================================================

def create_encoder():

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


# ============================================================
# CREATE PREPROCESSED DATASET
# ============================================================

def create_preprocessed_file():

    df = load_data().copy()


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    df = df.drop_duplicates().copy()


    # ========================================================
    # NORMALIZE CLASS
    # ========================================================

    df[TARGET_COL] = normalize_class(
        df[TARGET_COL]
    )


    # ========================================================
    # REMOVE INVALID CLASS VALUES
    # ========================================================

    df = df[
        df[TARGET_COL].isin(
            ["BAD", "Good"]
        )
    ].copy()


    # ========================================================
    # HANDLE NUMERICAL MISSING VALUES
    # ========================================================

    for column in NUMERIC_COLS:

        median_value = df[column].median()

        df[column] = (
            df[column]
            .fillna(median_value)
        )


    # ========================================================
    # HANDLE CATEGORICAL MISSING VALUES
    # ========================================================

    for column in CATEGORICAL_COLS:

        mode_values = df[column].mode()

        if len(mode_values) > 0:

            mode_value = mode_values.iloc[0]

            df[column] = (
                df[column]
                .fillna(mode_value)
            )

        else:

            df[column] = (
                df[column]
                .fillna("Unknown")
            )


    # ========================================================
    # IQR OUTLIER TREATMENT
    # ========================================================

    for column in NUMERIC_COLS:

        Q1 = df[column].quantile(0.25)

        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_fence = (
            Q1 - 1.5 * IQR
        )

        upper_fence = (
            Q3 + 1.5 * IQR
        )

        df[column] = (
            df[column]
            .clip(
                lower=lower_fence,
                upper=upper_fence
            )
        )


    # ========================================================
    # ONE-HOT ENCODE FRUIT
    # ========================================================

    encoder = create_encoder()

    encoded = encoder.fit_transform(
        df[CATEGORICAL_COLS]
    )

    encoded_columns = (
        encoder
        .get_feature_names_out(
            CATEGORICAL_COLS
        )
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_columns,
        index=df.index
    )


    # ========================================================
    # CLASS ENCODING
    # ========================================================

    df["Class_Encoded"] = (
        df[TARGET_COL]
        .map(CLASS_MAPPING)
    )


    # ========================================================
    # FINAL PREPROCESSED DATA
    #
    # Keep numerical values unscaled here.
    #
    # Reason:
    # Scaling will be performed AFTER train/test split
    # inside algorithms that require it.
    # ========================================================

    final_df = pd.concat(
        [
            df[NUMERIC_COLS],
            encoded_df,
            df[["Class_Encoded"]]
        ],
        axis=1
    )


    # ========================================================
    # SAVE FILE
    # ========================================================

    final_df.to_csv(
        PREPROCESSED_PATH,
        index=False
    )


    return final_df


# ============================================================
# FULL PREPROCESSING SUMMARY
# ============================================================

def preprocess_data():

    original_df = load_data()

    original_rows = int(
        original_df.shape[0]
    )


    original_columns = int(
        original_df.shape[1]
    )


    duplicate_count = int(
        original_df.duplicated().sum()
    )


    # Create final file

    final_df = create_preprocessed_file()


    # ========================================================
    # TRAIN TEST SPLIT FOR DISPLAY
    # ========================================================

    X = final_df.drop(
        columns=["Class_Encoded"]
    )

    y = final_df["Class_Encoded"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )


    # ========================================================
    # OUTLIER SUMMARY
    #
    # Calculated from cleaned original data.
    # ========================================================

    cleaned = original_df.drop_duplicates().copy()

    cleaned[TARGET_COL] = normalize_class(
        cleaned[TARGET_COL]
    )

    cleaned = cleaned[
        cleaned[TARGET_COL].isin(
            ["BAD", "Good"]
        )
    ].copy()


    outlier_summary = []


    for column in NUMERIC_COLS:

        cleaned[column] = (
            cleaned[column]
            .fillna(
                cleaned[column].median()
            )
        )

        Q1 = cleaned[column].quantile(
            0.25
        )

        Q3 = cleaned[column].quantile(
            0.75
        )

        IQR = Q3 - Q1

        lower_fence = (
            Q1 - 1.5 * IQR
        )

        upper_fence = (
            Q3 + 1.5 * IQR
        )

        outliers = (
            (cleaned[column] < lower_fence)
            |
            (cleaned[column] > upper_fence)
        ).sum()


        outlier_summary.append({

            "Feature":
                column,

            "Q1":
                round(Q1, 3),

            "Q3":
                round(Q3, 3),

            "IQR":
                round(IQR, 3),

            "Lower Fence":
                round(
                    lower_fence,
                    3
                ),

            "Upper Fence":
                round(
                    upper_fence,
                    3
                ),

            "Training Outliers":
                int(outliers),

            "Testing Outliers":
                0

        })


    # ========================================================
    # SCALING PREVIEWS
    # ========================================================

    numeric_data = final_df[
        NUMERIC_COLS
    ]


    minmax_scaler = MinMaxScaler()

    minmax_result = (
        minmax_scaler
        .fit_transform(
            numeric_data
        )
    )


    minmax_df = pd.DataFrame(
        minmax_result,
        columns=NUMERIC_COLS
    )


    standard_scaler = StandardScaler()

    standard_result = (
        standard_scaler
        .fit_transform(
            numeric_data
        )
    )


    standard_df = pd.DataFrame(
        standard_result,
        columns=NUMERIC_COLS
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "original_rows":
            original_rows,

        "original_columns":
            original_columns,

        "duplicate_count":
            duplicate_count,

        "cleaned_rows":
            int(final_df.shape[0]),

        "train_rows":
            int(X_train.shape[0]),

        "test_rows":
            int(X_test.shape[0]),

        "numeric_columns":
            NUMERIC_COLS,

        "categorical_columns":
            CATEGORICAL_COLS,

        "outlier_summary":
            outlier_summary,

        "minmax_preview":
            minmax_df
            .head(5)
            .round(3)
            .to_dict(
                orient="records"
            ),

        "standard_preview":
            standard_df
            .head(5)
            .round(3)
            .to_dict(
                orient="records"
            ),

        "onehot_columns":
            [
                column
                for column in final_df.columns
                if column.startswith("Fruit_")
            ],

        "onehot_preview":
            final_df[
                [
                    column
                    for column in final_df.columns
                    if column.startswith("Fruit_")
                ]
            ]
            .head(5)
            .to_dict(
                orient="records"
            ),

        "class_mapping":
            CLASS_MAPPING,

        "classes":
            ["BAD", "Good"],

        "final_features":
            list(
                X.columns
            ),

        "final_train_preview":
            X_train
            .head(5)
            .round(3)
            .to_dict(
                orient="records"
            ),

        "target_train_preview":
            y_train
            .head(10)
            .tolist(),

        "X_train_shape":
            list(X_train.shape),

        "X_test_shape":
            list(X_test.shape),

        "preprocessed_file":
            str(PREPROCESSED_PATH.name)

    }


# ============================================================
# WEB SUMMARY
# ============================================================

def get_preprocessing_summary():

    return preprocess_data()