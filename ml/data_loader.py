from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DATASET PATH
# ============================================================

DATA_PATH = (
    BASE_DIR / "FreshChain_Dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the Fresh Chain fruit quality dataset.
    """

    df = pd.read_csv(
        DATA_PATH
    )

    return df


# ============================================================
# DATA SUMMARY
# ============================================================

def get_data_summary():
    """
    Return basic information about the dataset.
    """

    df = load_data()

    summary = {

        "n_rows": int(
            df.shape[0]
        ),

        "n_cols": int(
            df.shape[1]
        ),

        "columns": list(
            df.columns
        ),

        "dtypes": {
            column: str(
                df[column].dtype
            )
            for column in df.columns
        },

        "missing_counts": {
            column: int(
                df[column].isna().sum()
            )
            for column in df.columns
        },

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "preview": (
            df.head(10)
            .to_dict(
                orient="records"
            )
        )
    }

    return summary