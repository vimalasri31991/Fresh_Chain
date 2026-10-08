from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.model_data import get_model_data
from ml.data_loader import load_preprocessed_data
from ml.model_utils import make_logistic


LINEAR_FEATURES = ["Temp", "Humid (%)", "Light (Fux)"]
LINEAR_TARGET = "CO2 (pmm)"


# ============================================================
# LINEAR REGRESSION  (Lasso = L1, Ridge = L2)
# ============================================================

def _regression_metrics(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)

    return {
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(mse ** 0.5),
        "mae": float(mean_absolute_error(y_test, pred)),
        "coefficients": {
            name: round(float(c), 4)
            for name, c in zip(LINEAR_FEATURES, model.coef_)
        },
        "zeroed": int(sum(abs(c) < 1e-8 for c in model.coef_)),
    }


def _linear_block():

    data = load_preprocessed_data()

    X = data[LINEAR_FEATURES]
    y = data[LINEAR_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return {
        "target": LINEAR_TARGET,
        "features": LINEAR_FEATURES,
        "alpha": 1.0,
        "baseline": _regression_metrics(
            LinearRegression(), X_train_s, X_test_s, y_train, y_test
        ),
        "l1": _regression_metrics(
            Lasso(alpha=1.0, random_state=42),
            X_train_s, X_test_s, y_train, y_test
        ),
        "l2": _regression_metrics(
            Ridge(alpha=1.0, random_state=42),
            X_train_s, X_test_s, y_train, y_test
        ),
    }


# ============================================================
# LOGISTIC REGRESSION  (L1 and L2)
# ============================================================

def _classification_metrics(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).tolist(),
        "coefficients": {
            name: round(float(c), 4)
            for name, c in zip(X_train_columns(X_train), model.coef_[0])
        },
        "zeroed": int((abs(model.coef_[0]) < 1e-8).sum()),
    }


def X_train_columns(X_train):
    return list(getattr(X_train, "columns", []))


def _logistic_block():

    X, y = get_model_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()

    import pandas as pd

    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns
    )

    return {
        "C": 1.0,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": list(X.columns),
        "baseline": _classification_metrics(
            make_logistic("l2", C=1e6), X_train_s, X_test_s, y_train, y_test
        ),
        "l1": _classification_metrics(
            make_logistic("l1", C=1.0), X_train_s, X_test_s, y_train, y_test
        ),
        "l2": _classification_metrics(
            make_logistic("l2", C=1.0), X_train_s, X_test_s, y_train, y_test
        ),
    }


# ============================================================
# PUBLIC
# ============================================================

def get_regularisation():

    logistic = _logistic_block()

    return {
        "linear": _linear_block(),
        "logistic": logistic,
        "train_rows": logistic["train_rows"],
        "test_rows": logistic["test_rows"],
    }