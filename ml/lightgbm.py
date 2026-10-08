import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from ml.model_data import get_model_data


BASE_DIR = Path(__file__).resolve().parent.parent

CHART_DIR = BASE_DIR / "static" / "charts"


def get_lightgbm():

    X, y = get_model_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ========================================================
    # BEFORE IMPROVEMENT
    # ========================================================

    before_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
        verbosity=-1
    )

    before_model.fit(
        X_train,
        y_train
    )

    before_prediction = before_model.predict(
        X_test
    )

    # ========================================================
    # AFTER IMPROVEMENT
    # ========================================================

    after_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=20,
        max_depth=5,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )

    after_model.fit(
        X_train,
        y_train
    )

    after_prediction = after_model.predict(
        X_test
    )

    # ========================================================
    # METRICS
    # ========================================================

    before_accuracy = accuracy_score(
        y_test,
        before_prediction
    )

    after_accuracy = accuracy_score(
        y_test,
        after_prediction
    )

    before_precision = precision_score(
        y_test,
        before_prediction,
        zero_division=0
    )

    after_precision = precision_score(
        y_test,
        after_prediction,
        zero_division=0
    )

    before_recall = recall_score(
        y_test,
        before_prediction,
        zero_division=0
    )

    after_recall = recall_score(
        y_test,
        after_prediction,
        zero_division=0
    )

    before_f1 = f1_score(
        y_test,
        before_prediction,
        zero_division=0
    )

    after_f1 = f1_score(
        y_test,
        after_prediction,
        zero_division=0
    )

    after_cm = confusion_matrix(
        y_test,
        after_prediction
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance = {
        feature: round(
            float(importance),
            4
        )
        for feature, importance in zip(
            X.columns,
            after_model.feature_importances_
        )
    }

    # ========================================================
    # CHART
    # ========================================================

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        after_cm
    )

    plt.title(
        "LightGBM Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    for i in range(len(after_cm)):

        for j in range(len(after_cm[i])):

            plt.text(
                j,
                i,
                after_cm[i][j],
                ha="center",
                va="center"
            )

    plt.xticks(
        [0, 1],
        ["Bad", "Good"]
    )

    plt.yticks(
        [0, 1],
        ["Bad", "Good"]
    )

    plt.tight_layout()

    chart_path = CHART_DIR / "lightgbm_confusion_matrix.png"

    plt.savefig(
        chart_path
    )

    plt.close()

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "train_rows": len(X_train),

        "test_rows": len(X_test),

        "before_accuracy":
            round(before_accuracy * 100, 2),

        "after_accuracy":
            round(after_accuracy * 100, 2),

        "before_precision":
            round(before_precision * 100, 2),

        "after_precision":
            round(after_precision * 100, 2),

        "before_recall":
            round(before_recall * 100, 2),

        "after_recall":
            round(after_recall * 100, 2),

        "before_f1":
            round(before_f1 * 100, 2),

        "after_f1":
            round(after_f1 * 100, 2),

        "n_estimators":
            after_model.n_estimators,

        "learning_rate":
            after_model.learning_rate,

        "num_leaves":
            after_model.num_leaves,

        "max_depth":
            after_model.max_depth,

        "feature_importance":
            feature_importance,

        "confusion_matrix":
            after_cm.tolist(),

        "chart":
            "charts/lightgbm_confusion_matrix.png"
    }