from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from ml.data_loader import load_preprocessed_data


BASE_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = BASE_DIR / "static" / "charts"


def get_cost_complexity():

    data = load_preprocessed_data()

    target = "Class_Encoded"

    X = data.drop(columns=[target])
    y = data[target]

    # -------------------------------------------------
    # First split: development + test
    # -------------------------------------------------
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # -------------------------------------------------
    # Second split: training + validation
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.20,
        random_state=42,
        stratify=y_dev
    )

    # -------------------------------------------------
    # Get pruning path
    # -------------------------------------------------
    base_tree = DecisionTreeClassifier(
        random_state=42
    )

    pruning_path = base_tree.cost_complexity_pruning_path(
        X_train,
        y_train
    )

    ccp_alphas = pruning_path.ccp_alphas

    # Remove duplicate alpha values
    ccp_alphas = sorted(set(ccp_alphas))

    # Avoid testing an unnecessarily large number of trees
    if len(ccp_alphas) > 30:
        indexes = [
            int(i)
            for i in
            __import__("numpy").linspace(
                0,
                len(ccp_alphas) - 1,
                30
            )
        ]

        ccp_alphas = [
            ccp_alphas[i]
            for i in indexes
        ]

    # -------------------------------------------------
    # Find best alpha using VALIDATION data
    # -------------------------------------------------
    best_alpha = ccp_alphas[0]
    best_validation_accuracy = -1

    for alpha in ccp_alphas:

        tree = DecisionTreeClassifier(
            random_state=42,
            ccp_alpha=alpha
        )

        tree.fit(X_train, y_train)

        validation_pred = tree.predict(X_val)

        validation_accuracy = accuracy_score(
            y_val,
            validation_pred
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_alpha = alpha

    # -------------------------------------------------
    # Train final model using complete development set
    # -------------------------------------------------
    final_tree = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=best_alpha
    )

    final_tree.fit(X_dev, y_dev)

    # -------------------------------------------------
    # Test final model ONCE
    # -------------------------------------------------
    train_pred = final_tree.predict(X_dev)
    test_pred = final_tree.predict(X_test)

    train_accuracy = accuracy_score(
        y_dev,
        train_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    precision = precision_score(
        y_test,
        test_pred,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_pred,
        pos_label=1,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=[0, 1]
    )

    # -------------------------------------------------
    # Generate confusion matrix chart
    # -------------------------------------------------
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)

    plt.title("Cost-Complexity Pruning Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(
                j,
                i,
                cm[i][j],
                ha="center",
                va="center"
            )

    plt.xticks([0, 1], ["BAD", "Good"])
    plt.yticks([0, 1], ["BAD", "Good"])

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "cost_complexity_confusion.png"
    )

    plt.close()

    # -------------------------------------------------
    # Return data expected by app.py/template
    # -------------------------------------------------
    return {
        "total_rows": len(X),
        "train_rows": len(X_dev),
        "test_rows": len(X_test),

        "features": list(X.columns),
        "target": target,

        "best_alpha": round(
            float(best_alpha),
            6
        ),

        "validation_accuracy": round(
            float(best_validation_accuracy) * 100,
            2
        ),

        "train_accuracy": round(
            float(train_accuracy) * 100,
            2
        ),

        "test_accuracy": round(
            float(test_accuracy) * 100,
            2
        ),

        "precision": round(
            float(precision) * 100,
            2
        ),

        "recall": round(
            float(recall) * 100,
            2
        ),

        "f1": round(
            float(f1) * 100,
            2
        ),

        "confusion_matrix": cm.tolist(),

        "depth": final_tree.get_depth(),

        "leaves": final_tree.get_n_leaves(),

        "candidate_count": len(ccp_alphas)
    }