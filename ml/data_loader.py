from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "FreshChain_Dataset.csv"

PREPROCESSED_PATH = BASE_DIR / "preprocessed_data.csv"


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD PREPROCESSED DATASET
# ============================================================

def load_preprocessed_data():

    if not PREPROCESSED_PATH.exists():

        from ml.preprocessing import create_preprocessed_file

        create_preprocessed_file()

    if not PREPROCESSED_PATH.exists():

        raise FileNotFoundError(
            "Preprocessed dataset could not be created."
        )

    return pd.read_csv(
        PREPROCESSED_PATH
    )


# ============================================================
# DATA SUMMARY
# ============================================================

def get_data_summary():

    df = load_data()

    missing = df.isnull().sum()

    missing_summary = []

    for column in df.columns:

        missing_summary.append({

            "column":
                column,

            "dtype":
                str(df[column].dtype),

            "missing":
                int(missing[column]),

            "unique":
                int(df[column].nunique())

        })


    return {

        "rows":
            int(df.shape[0]),

        "columns":
            int(df.shape[1]),

        "column_names":
            list(df.columns),

        "missing_total":
            int(df.isnull().sum().sum()),

        "duplicate_rows":
            int(df.duplicated().sum()),

        "preview":
            df.head(10).to_dict(
                orient="records"
            ),

        "summary":
            missing_summary
    }