from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ml.data_loader import load_preprocessed_data


BASE_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = BASE_DIR / "static" / "charts"


def prepare_data():
    data = load_preprocessed_data()

    target = "Class_Encoded"

    X = data.drop(columns=[target])
    y = data[target]

    return X, y


def generate_confusion_chart(cm, filename, title):
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.xticks([0, 1], ["BAD", "Good"])
    plt.yticks([0, 1], ["BAD", "Good"])

    plt.tight_layout()
    plt.savefig(CHART_DIR / filename)
    plt.close()


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

    return {
        "name": name,
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(
            y_test,
            test_pred,
            pos_label=1,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            test_pred,
            pos_label=1,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            test_pred,
            pos_label=1,
            zero_division=0
        ),
        "confusion_matrix": cm
    }


def get_logistic_regression():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    results = []

    # -------------------------------------------------
    # 1. Unscaled Logistic Regression
    # -------------------------------------------------
    unscaled_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    unscaled = evaluate_model(
        "Unscaled",
        unscaled_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(unscaled)

    generate_confusion_chart(
        unscaled["confusion_matrix"],
        "logistic_unscaled_confusion.png",
        "Logistic Regression - Unscaled"
    )

    # -------------------------------------------------
    # 2. Standard Scaled Logistic Regression
    # -------------------------------------------------
    standard_scaler = StandardScaler()

    X_train_standard = standard_scaler.fit_transform(X_train)
    X_test_standard = standard_scaler.transform(X_test)

    standard_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    standard = evaluate_model(
        "Standard Scaler",
        standard_model,
        X_train_standard,
        X_test_standard,
        y_train,
        y_test
    )

    results.append(standard)

    generate_confusion_chart(
        standard["confusion_matrix"],
        "logistic_standard_confusion.png",
        "Logistic Regression - Standard Scaler"
    )

    # -------------------------------------------------
    # 3. Min-Max Scaled Logistic Regression
    # -------------------------------------------------
    minmax_scaler = MinMaxScaler()

    X_train_minmax = minmax_scaler.fit_transform(X_train)
    X_test_minmax = minmax_scaler.transform(X_test)

    minmax_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    minmax = evaluate_model(
        "Min-Max Scaler",
        minmax_model,
        X_train_minmax,
        X_test_minmax,
        y_train,
        y_test
    )

    results.append(minmax)

    generate_confusion_chart(
        minmax["confusion_matrix"],
        "logistic_minmax_confusion.png",
        "Logistic Regression - Min-Max Scaler"
    )

    # -------------------------------------------------
    # Find Best Model
    # -------------------------------------------------
    best = max(
        results,
        key=lambda result: result["test_accuracy"]
    )

    comparison = []

    for result in results:
        comparison.append({
            "name": result["name"],
            "train_accuracy": round(
                float(result["train_accuracy"]) * 100, 2
            ),
            "test_accuracy": round(
                float(result["test_accuracy"]) * 100, 2
            ),
            "precision": round(
                float(result["precision"]) * 100, 2
            ),
            "recall": round(
                float(result["recall"]) * 100, 2
            ),
            "f1": round(
                float(result["f1"]) * 100, 2
            )
        })

    return {
        "total_rows": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),

        "features": list(X.columns),
        "target": "Class_Encoded",

        "classes": ["BAD", "Good"],

        "best_model": best["name"],

        "best_accuracy": round(
            float(best["test_accuracy"]) * 100, 2
        ),

        "best_precision": round(
            float(best["precision"]) * 100, 2
        ),

        "best_recall": round(
            float(best["recall"]) * 100, 2
        ),

        "best_f1": round(
            float(best["f1"]) * 100, 2
        ),

        "best_confusion_matrix":
            best["confusion_matrix"].tolist(),

        "comparison": comparison
    }


# Compatibility function
def logistic_regression():
    return get_logistic_regression()