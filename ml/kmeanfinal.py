import os

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "FreshChain_Dataset.csv"
)

FEATURES = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]


def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    X = df[FEATURES]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    return df, X_scaled


def perform_kmeans(k):

    df, X = load_data()

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X
    )

    result_df = df.copy()

    result_df["Cluster"] = labels

    return (
        result_df,
        model,
        X
    )


def kmeanfinal(k):

    return perform_kmeans(k)


def get_kmeans_result(k):

    return perform_kmeans(k)