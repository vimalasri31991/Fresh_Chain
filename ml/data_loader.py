from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "FreshChain_Dataset.csv"


# ============================================================
# LOAD DATASET
# ============================================================

def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# DATA SUMMARY
# ============================================================

def get_data_summary():

    df = load_data()

    missing = df.isnull().sum()

    missing_summary = []

    for column in df.columns:

        missing_summary.append({

            "column": column,

            "dtype": str(
                df[column].dtype
            ),

            "missing": int(
                missing[column]
            ),

            "unique": int(
                df[column].nunique()
            )
        })


    return {

        "rows": int(
            df.shape[0]
        ),

        "columns": int(
            df.shape[1]
        ),

        "column_names": list(
            df.columns
        ),

        "missing_total": int(
            df.isnull().sum().sum()
        ),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "preview": (
            df.head(10)
            .to_dict(
                orient="records"
            )
        ),

        "summary": missing_summary
    }