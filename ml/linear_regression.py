from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split


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
# FEATURES AND TARGET
# ============================================================

FEATURES = [
    "Temp",
    "Humid (%)",
    "Light (Fux)"
]

TARGET = "CO2 (pmm)"


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

def prepare_data():

    from ml.data_loader import load_data

    df = load_data()

    data = df[
        FEATURES + [TARGET]
    ].dropna()

    X = data[FEATURES]

    y = data[TARGET]

    return X, y


# ============================================================
# GENERATE LINEAR REGRESSION CHARTS
# ============================================================

def generate_linear_charts(
    y_test,
    y_pred,
    residuals
):

    # --------------------------------------------------------
    # Actual vs Predicted
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.55
    )

    minimum = min(
        y_test.min(),
        y_pred.min()
    )

    maximum = max(
        y_test.max(),
        y_pred.max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    ax.set_title(
        "Actual vs Predicted CO₂"
    )

    ax.set_xlabel(
        "Actual CO₂ (pmm)"
    )

    ax.set_ylabel(
        "Predicted CO₂ (pmm)"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        CHART_DIR /
        "linear_actual_vs_predicted.png",
        dpi=140,
        bbox_inches="tight"
    )

    plt.close(fig)


    # --------------------------------------------------------
    # Residual Histogram
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.hist(
        residuals,
        bins=40,
        edgecolor="white"
    )

    ax.axvline(
        0,
        linestyle="--"
    )

    ax.set_title(
        "Linear Regression Residual Distribution"
    )

    ax.set_xlabel(
        "Residual"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        CHART_DIR /
        "linear_residuals.png",
        dpi=140,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# LINEAR REGRESSION
# ============================================================

def get_linear_regression():

    X, y = prepare_data()


    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = mse ** 0.5

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # RESIDUALS
    # --------------------------------------------------------

    residuals = y_test - y_pred


    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    generate_linear_charts(
        y_test,
        y_pred,
        residuals
    )


    # --------------------------------------------------------
    # COEFFICIENTS
    # --------------------------------------------------------

    coefficients = {

        feature: round(
            float(coefficient),
            4
        )

        for feature, coefficient

        in zip(
            FEATURES,
            model.coef_
        )
    }


    # --------------------------------------------------------
    # EQUATION
    # --------------------------------------------------------

    equation = (
        f"CO2 = "
        f"{model.intercept_:.3f}"
        f" + "
        f"({model.coef_[0]:.3f} × Temp)"
        f" + "
        f"({model.coef_[1]:.3f} × Humidity)"
        f" + "
        f"({model.coef_[2]:.3f} × Light)"
    )


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "total_rows": int(
            len(X)
        ),

        "train_rows": int(
            len(X_train)
        ),

        "test_rows": int(
            len(X_test)
        ),

        "features": FEATURES,

        "target": TARGET,

        "mse": round(
            float(mse),
            4
        ),

        "rmse": round(
            float(rmse),
            4
        ),

        "mae": round(
            float(mae),
            4
        ),

        "r2": round(
            float(r2),
            4
        ),

        "intercept": round(
            float(model.intercept_),
            4
        ),

        "coefficients": coefficients,

        "equation": equation
    }