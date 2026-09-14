import os

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)


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


def evaluate_kmeans(k):

    df, X = load_data()

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X
    )

    silhouette = silhouette_score(
        X,
        labels,
        sample_size=min(
            2000,
            len(X)
        ),
        random_state=42
    )

    davies_bouldin = davies_bouldin_score(
        X,
        labels
    )

    cluster_counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {

        "k": k,

        "inertia": round(
            model.inertia_,
            4
        ),

        "silhouette_score": round(
            silhouette,
            4
        ),

        "davies_bouldin_score": round(
            davies_bouldin,
            4
        ),

        "clusters": cluster_counts
    }


def kmeansperformance(k):

    return evaluate_kmeans(k)


def get_kmeans_performance(k):

    return evaluate_kmeans(k)