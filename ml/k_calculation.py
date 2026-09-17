import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return df, X_scaled


def calculate_k(selected_k=None):

    df, X = load_data()

    k_values = list(range(2, 11))

    wcss = []

    silhouette_scores = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        wcss.append(model.inertia_)

        score = silhouette_score(
            X,
            labels,
            sample_size=min(2000, len(X)),
            random_state=42
        )

        silhouette_scores.append(score)

    chart_folder = os.path.join(
        BASE_DIR,
        "static",
        "charts"
    )

    os.makedirs(
        chart_folder,
        exist_ok=True
    )

    elbow_path = os.path.join(
        chart_folder,
        "kmeans_elbow.png"
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        k_values,
        wcss,
        marker="o"
    )

    if selected_k is not None:
        selected_index = k_values.index(selected_k)

        plt.scatter(
            selected_k,
            wcss[selected_index],
            color="red",
            s=100,
            zorder=5
        )

        plt.axvline(
            selected_k,
            color="red",
            linestyle="--",
            alpha=0.6
        )

        plt.annotate(
            f"Selected K = {selected_k}",
            (selected_k, wcss[selected_index]),
            xytext=(10, 10),
            textcoords="offset points",
            color="red",
            fontweight="bold"
        )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")

    if selected_k is not None:
        plt.title(f"Elbow Method - Selected K = {selected_k}")
    else:
        plt.title("Elbow Method")

    plt.xticks(k_values)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        elbow_path,
        dpi=100
    )

    plt.close()

    silhouette_path = os.path.join(
        chart_folder,
        "kmeans_silhouette.png"
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        k_values,
        silhouette_scores,
        marker="o"
    )

    if selected_k is not None:
        selected_index = k_values.index(selected_k)

        plt.scatter(
            selected_k,
            silhouette_scores[selected_index],
            color="red",
            s=100,
            zorder=5
        )

        plt.axvline(
            selected_k,
            color="red",
            linestyle="--",
            alpha=0.6
        )

        plt.annotate(
            f"Selected K = {selected_k}",
            (selected_k, silhouette_scores[selected_index]),
            xytext=(10, 10),
            textcoords="offset points",
            color="red",
            fontweight="bold"
        )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")

    if selected_k is not None:
        plt.title(f"Silhouette Method - Selected K = {selected_k}")
    else:
        plt.title("Silhouette Method")

    plt.xticks(k_values)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        silhouette_path,
        dpi=100
    )

    plt.close()

    return {
        "k_values": k_values,
        "wcss": wcss,
        "silhouette_scores": silhouette_scores,
        "elbow_chart": "charts/kmeans_elbow.png",
        "silhouette_chart": "charts/kmeans_silhouette.png"
    }


def k_all():

    return calculate_k()


def get_k_calculation():

    return calculate_k()