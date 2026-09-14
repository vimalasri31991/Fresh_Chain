import pandas as pd

from ml.data_loader import load_preprocessed_data


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]


FRUIT_FEATURES = [
    "Fruit_Banana",
    "Fruit_Orange",
    "Fruit_Pineapple",
    "Fruit_Tomato"
]


FEATURES = (
    NUMERIC_FEATURES
    + FRUIT_FEATURES
)


TARGET = "Class_Encoded"


# ============================================================
# LOAD MODEL DATA
# ============================================================

def get_model_data():

    df = load_preprocessed_data()

    X = df[FEATURES].copy()

    y = df[TARGET].copy()

    return X, y


# ============================================================
# FEATURE NAMES
# ============================================================

def get_feature_names():

    return FEATURES


# ============================================================
# CLASS NAMES
# ============================================================

def get_class_names():

    return [
        "BAD",
        "Good"
    ]