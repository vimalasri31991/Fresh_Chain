from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ml.data_loader import load_preprocessed_data


BASE_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = BASE_DIR / "static" / "charts"

FEATURES = ["Temp", "Humid (%)", "Light (Fux)"]
TARGET = "CO2 (pmm)"


def prepare_data():
    data = load_preprocessed_data()

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    return X, y


def generate_linear_charts(y_test, y_pred, residuals):
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Actual vs Predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual CO2")
    plt.ylabel("Predicted CO2")
    plt.title("Linear Regression - Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "linear_actual_vs_predicted.png")
    plt.close()

    # Residual plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Predicted CO2")
    plt.ylabel("Residual")
    plt.title("Linear Regression - Residual Plot")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "linear_residuals.png")
    plt.close()


def get_linear_regression():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    residuals = y_test - y_pred

    generate_linear_charts(y_test, y_pred, residuals)

    coefficients = {}

    for feature, coefficient in zip(FEATURES, model.coef_):
        coefficients[feature] = round(float(coefficient), 4)

    equation = (
        f"CO2 = {model.intercept_:.4f}"
        f" + ({model.coef_[0]:.4f} × Temp)"
        f" + ({model.coef_[1]:.4f} × Humid (%))"
        f" + ({model.coef_[2]:.4f} × Light (Fux))"
    )

    return {
        "total_rows": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),

        "features": FEATURES,
        "target": TARGET,

        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),

        "intercept": round(float(model.intercept_), 4),
        "coefficients": coefficients,
        "equation": equation
    }


# Compatibility function
def linear_regression():
    return get_linear_regression()