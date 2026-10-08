import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
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


def get_hierarchical_clustering():

    df, X = prepare_data()

    # Default number of clusters
    n_clusters = 2

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )

    labels = model.fit_predict(X)

    result_df = df.copy()

    result_df["Cluster"] = labels

    unique_labels = sorted(
        result_df["Cluster"].unique()
    )

    cluster_counts = (
        result_df["Cluster"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if len(unique_labels) > 1:

        silhouette = silhouette_score(
            X,
            labels
        )

        davies_bouldin = davies_bouldin_score(
            X,
            labels
        )

    else:

        silhouette = 0.0
        davies_bouldin = 0.0


    return {

        "method":
            "Hierarchical Clustering",

        "algorithm":
            "Agglomerative Clustering",

        "linkage":
            "Ward",

        "n_clusters":
            n_clusters,

        "total_rows":
            len(df),

        "features":
            FEATURES,

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
def hierarchical_clustering():

    return get_hierarchical_clustering()