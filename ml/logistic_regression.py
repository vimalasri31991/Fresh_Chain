from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)


# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHART_DIR = BASE_DIR / "static" / "charts"

CHART_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]

TARGET = "Class"


# ============================================================
# CLASS LABELS
# ============================================================

BAD_CLASS = "BAD"
GOOD_CLASS = "Good"

CLASS_LABELS = [
    BAD_CLASS,
    GOOD_CLASS
]


# ============================================================
# LOAD DATA
# ============================================================

def prepare_data():

    from ml.data_loader import load_data

    df = load_data()

    data = df[
        FEATURES + [TARGET]
    ].dropna()

    X = data[FEATURES]

    # --------------------------------------------------------
    # IMPORTANT:
    # Both "Bad" and "BAD" are converted to "BAD".
    # "Good" remains "Good".
    # --------------------------------------------------------

    y = (
        data[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "bad": BAD_CLASS,
            "good": GOOD_CLASS
        })
    )

    return X, y


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

def train_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    model = LogisticRegression(
        max_iter=1000
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    train_pred = model.predict(
        X_train
    )

    test_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        test_pred,
        pos_label=GOOD_CLASS,
        zero_division=0
    )

    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    recall = recall_score(
        y_test,
        test_pred,
        pos_label=GOOD_CLASS,
        zero_division=0
    )

    # --------------------------------------------------------
    # F1 SCORE
    # --------------------------------------------------------

    f1 = f1_score(
        y_test,
        test_pred,
        pos_label=GOOD_CLASS,
        zero_division=0
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=CLASS_LABELS
    )

    return {

        "model": model,

        "train_accuracy":
            train_accuracy,

        "test_accuracy":
            test_accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "confusion_matrix":
            cm
    }


# ============================================================
# CONFUSION MATRIX CHART
# ============================================================

def generate_confusion_chart(
    matrix,
    filename,
    title
):

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        matrix,
        interpolation="nearest"
    )

    fig.colorbar(
        image,
        ax=ax
    )

    labels = [
        BAD_CLASS,
        GOOD_CLASS
    ]

    ax.set_xticks(
        range(2)
    )

    ax.set_yticks(
        range(2)
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_title(
        title
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center"
            )

    fig.tight_layout()

    fig.savefig(
        CHART_DIR / filename,
        dpi=140,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def get_logistic_regression():

    X, y = prepare_data()

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    results = {}


    # ========================================================
    # 1. UNSCALED
    # ========================================================

    unscaled = train_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results["Unscaled"] = unscaled


    # ========================================================
    # 2. STANDARD SCALER
    # ========================================================

    standard_scaler = StandardScaler()

    X_train_std = (
        standard_scaler
        .fit_transform(X_train)
    )

    X_test_std = (
        standard_scaler
        .transform(X_test)
    )

    standard = train_model(
        X_train_std,
        X_test_std,
        y_train,
        y_test
    )

    results["StandardScaler"] = standard


    # ========================================================
    # 3. MIN-MAX SCALER
    # ========================================================

    minmax_scaler = MinMaxScaler()

    X_train_mm = (
        minmax_scaler
        .fit_transform(X_train)
    )

    X_test_mm = (
        minmax_scaler
        .transform(X_test)
    )

    minmax = train_model(
        X_train_mm,
        X_test_mm,
        y_train,
        y_test
    )

    results["MinMaxScaler"] = minmax


    # ========================================================
    # GENERATE CONFUSION MATRICES
    # ========================================================

    generate_confusion_chart(
        unscaled["confusion_matrix"],
        "logistic_unscaled_confusion.png",
        "Logistic Regression - Unscaled"
    )

    generate_confusion_chart(
        standard["confusion_matrix"],
        "logistic_standard_confusion.png",
        "Logistic Regression - StandardScaler"
    )

    generate_confusion_chart(
        minmax["confusion_matrix"],
        "logistic_minmax_confusion.png",
        "Logistic Regression - MinMaxScaler"
    )


    # ========================================================
    # FIND BEST MODEL
    # ========================================================

    best_name = max(
        results,
        key=lambda name:
        results[name]["test_accuracy"]
    )

    best = results[best_name]


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison = []

    for name, result in results.items():

        comparison.append({

            "name":
                name,

            "train_accuracy":
                round(
                    result["train_accuracy"] * 100,
                    2
                ),

            "test_accuracy":
                round(
                    result["test_accuracy"] * 100,
                    2
                ),

            "precision":
                round(
                    result["precision"] * 100,
                    2
                ),

            "recall":
                round(
                    result["recall"] * 100,
                    2
                ),

            "f1":
                round(
                    result["f1"] * 100,
                    2
                )
        })


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "total_rows":
            int(len(X)),

        "train_rows":
            int(len(X_train)),

        "test_rows":
            int(len(X_test)),

        "features":
            FEATURES,

        "target":
            TARGET,

        "classes":
            CLASS_LABELS,

        "comparison":
            comparison,

        "best_model":
            best_name,

        "best_accuracy":
            round(
                best["test_accuracy"] * 100,
                2
            ),

        "best_confusion_matrix":
            best["confusion_matrix"].tolist(),

        "best_precision":
            round(
                best["precision"] * 100,
                2
            ),

        "best_recall":
            round(
                best["recall"] * 100,
                2
            ),

        "best_f1":
            round(
                best["f1"] * 100,
                2
            )
    }