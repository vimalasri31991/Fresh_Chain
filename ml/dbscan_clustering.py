import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

from ml.data_loader import load_data


FEATURES = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]


def prepare_data():

    df = load_data()

    X = df[FEATURES].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return df, X_scaled


def get_dbscan_clustering():

    df, X = prepare_data()

    # DBSCAN parameters
    eps = 0.8
    min_samples = 5

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X)

    result_df = df.copy()

    result_df["Cluster"] = labels


    # Count clusters
    cluster_counts = (
        result_df["Cluster"]
        .value_counts()
        .sort_index()
        .to_dict()
    )


    # Noise points are represented by -1
    noise_count = int(
        (labels == -1).sum()
    )


    # Valid clusters excluding noise
    valid_labels = set(labels)

    valid_labels.discard(-1)


    if len(valid_labels) >= 2:

        valid_mask = labels != -1

        X_valid = X[valid_mask]

        labels_valid = labels[valid_mask]

        silhouette = silhouette_score(
            X_valid,
            labels_valid
        )

        davies_bouldin = davies_bouldin_score(
            X_valid,
            labels_valid
        )

    else:

        silhouette = 0.0
        davies_bouldin = 0.0


    return {

        "method":
            "DBSCAN",

        "algorithm":
            "Density-Based Spatial Clustering",

        "eps":
            eps,

        "min_samples":
            min_samples,

        "total_rows":
            len(df),

        "features":
            FEATURES,

        "cluster_count":
            len(valid_labels),

        "noise_count":
            noise_count,

        "silhouette_score":
            round(
                float(silhouette),
                4
            ),

        "davies_bouldin_score":
            round(
                float(davies_bouldin),
                4
            ),

        "clusters":
            {
                str(key): int(value)
                for key, value
                in cluster_counts.items()
            }

    }


# Compatibility function
def dbscan_clustering():

    return get_dbscan_clustering()