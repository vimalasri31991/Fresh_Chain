import pandas as pd

from ml.data_loader import load_data


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_MAPPING = {
    "Bad": 0,
    "Good": 1
}


# ============================================================
# NORMALIZE CLASS
# ============================================================

def normalize_class(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "bad": "Bad",
            "good": "Good"
        })
    )


# ============================================================
# ENCODE CLASS
# ============================================================

def encode_class():

    df = load_data()

    result = normalize_class(
        df["Class"]
    )

    encoded = result.map(
        CLASS_MAPPING
    )

    return {

        "mapping":
            CLASS_MAPPING,

        "original":
            result.head(10).tolist(),

        "encoded":
            encoded.head(10).tolist()
    }