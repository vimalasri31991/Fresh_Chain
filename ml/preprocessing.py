import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder
)

from ml.data_loader import load_data


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
# MAIN PREPROCESSING
# ============================================================

def preprocess_data():

    df = load_data().copy()


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    duplicate_count = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates()


    # ========================================================
    # NORMALIZE CLASS LABELS
    # ========================================================

    df[TARGET_COL] = normalize_class(
        df[TARGET_COL]
    )


    # ========================================================
    # REMOVE INVALID CLASS ROWS
    # ========================================================

    df = df[
        df[TARGET_COL].isin(
            ["BAD", "Good"]
        )
    ].copy()


    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df[TARGET_COL]
    )

    train_df = train_df.copy()
    test_df = test_df.copy()


    # ========================================================
    # MISSING NUMERICAL VALUES
    # ========================================================

    for column in NUMERIC_COLS:

        median_value = (
            train_df[column]
            .median()
        )

        train_df[column] = (
            train_df[column]
            .fillna(median_value)
        )

        test_df[column] = (
            test_df[column]
            .fillna(median_value)
        )


    # ========================================================
    # MISSING CATEGORICAL VALUES
    # ========================================================

    for column in CATEGORICAL_COLS:

        mode_values = (
            train_df[column]
            .mode()
        )

        if len(mode_values) > 0:

            mode_value = mode_values.iloc[0]

            train_df[column] = (
                train_df[column]
                .fillna(mode_value)
            )

            test_df[column] = (
                test_df[column]
                .fillna(mode_value)
            )


    # ========================================================
    # IQR OUTLIER HANDLING
    # ========================================================

    outlier_summary = []


    for column in NUMERIC_COLS:

        Q1 = train_df[column].quantile(
            0.25
        )

        Q3 = train_df[column].quantile(
            0.75
        )

        IQR = Q3 - Q1

        lower_fence = (
            Q1 - 1.5 * IQR
        )

        upper_fence = (
            Q3 + 1.5 * IQR
        )


        train_outliers = (
            (
                train_df[column]
                < lower_fence
            )
            |
            (
                train_df[column]
                > upper_fence
            )
        ).sum()


        test_outliers = (
            (
                test_df[column]
                < lower_fence
            )
            |
            (
                test_df[column]
                > upper_fence
            )
        ).sum()


        train_df[column] = (
            train_df[column]
            .clip(
                lower=lower_fence,
                upper=upper_fence
            )
        )


        test_df[column] = (
            test_df[column]
            .clip(
                lower=lower_fence,
                upper=upper_fence
            )
        )


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
                int(train_outliers),

            "Testing Outliers":
                int(test_outliers)

        })


    # ========================================================
    # MIN-MAX SCALING
    # ========================================================

    minmax_scaler = MinMaxScaler()

    train_minmax = (
        minmax_scaler
        .fit_transform(
            train_df[NUMERIC_COLS]
        )
    )

    test_minmax = (
        minmax_scaler
        .transform(
            test_df[NUMERIC_COLS]
        )
    )


    train_minmax_df = pd.DataFrame(
        train_minmax,
        columns=NUMERIC_COLS,
        index=train_df.index
    )

    test_minmax_df = pd.DataFrame(
        test_minmax,
        columns=NUMERIC_COLS,
        index=test_df.index
    )


    # ========================================================
    # STANDARD SCALING
    # ========================================================

    standard_scaler = StandardScaler()

    train_standard = (
        standard_scaler
        .fit_transform(
            train_df[NUMERIC_COLS]
        )
    )

    test_standard = (
        standard_scaler
        .transform(
            test_df[NUMERIC_COLS]
        )
    )


    train_standard_df = pd.DataFrame(
        train_standard,
        columns=NUMERIC_COLS,
        index=train_df.index
    )

    test_standard_df = pd.DataFrame(
        test_standard,
        columns=NUMERIC_COLS,
        index=test_df.index
    )


    # ========================================================
    # ONE HOT ENCODING
    # ========================================================

    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )


    train_ohe = (
        ohe.fit_transform(
            train_df[CATEGORICAL_COLS]
        )
    )

    test_ohe = (
        ohe.transform(
            test_df[CATEGORICAL_COLS]
        )
    )


    ohe_columns = (
        ohe.get_feature_names_out(
            CATEGORICAL_COLS
        )
    )


    train_ohe_df = pd.DataFrame(
        train_ohe,
        columns=ohe_columns,
        index=train_df.index
    )

    test_ohe_df = pd.DataFrame(
        test_ohe,
        columns=ohe_columns,
        index=test_df.index
    )


    # ========================================================
    # TARGET ENCODING
    # ========================================================

    class_mapping = {
        "BAD": 0,
        "Good": 1
    }


    y_train = (
        normalize_class(
            train_df[TARGET_COL]
        )
        .map(class_mapping)
    )

    y_test = (
        normalize_class(
            test_df[TARGET_COL]
        )
        .map(class_mapping)
    )


    # ========================================================
    # FINAL FEATURES
    # ========================================================

    X_train = pd.concat(
        [
            train_standard_df,
            train_ohe_df
        ],
        axis=1
    )


    X_test = pd.concat(
        [
            test_standard_df,
            test_ohe_df
        ],
        axis=1
    )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "original_rows":
            int(load_data().shape[0]),

        "original_columns":
            int(load_data().shape[1]),

        "duplicate_count":
            duplicate_count,

        "cleaned_rows":
            int(df.shape[0]),

        "train_rows":
            int(train_df.shape[0]),

        "test_rows":
            int(test_df.shape[0]),

        "numeric_columns":
            NUMERIC_COLS,

        "categorical_columns":
            CATEGORICAL_COLS,

        "outlier_summary":
            outlier_summary,

        "minmax_preview":
            train_minmax_df
            .head(5)
            .round(3)
            .to_dict(
                orient="records"
            ),

        "standard_preview":
            train_standard_df
            .head(5)
            .round(3)
            .to_dict(
                orient="records"
            ),

        "onehot_columns":
            list(ohe_columns),

        "onehot_preview":
            train_ohe_df
            .head(5)
            .to_dict(
                orient="records"
            ),

        "class_mapping":
            class_mapping,

        "classes":
            ["BAD", "Good"],

        "final_features":
            list(X_train.columns),

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
            list(X_test.shape)
    }


# ============================================================
# WEB SUMMARY
# ============================================================

def get_preprocessing_summary():

    return preprocess_data()