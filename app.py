from flask import Flask, render_template

from ml.data_loader import get_data_summary
from ml.eda import get_eda_summary, generate_eda_charts
from ml.preprocessing import get_preprocessing_summary
from ml.linear_regression import get_linear_regression
from ml.logistic_regression import get_logistic_regression


app = Flask(__name__)


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="dashboard"
    )


# ============================================================
# DATA LOADING
# ============================================================

@app.route("/data-loading")
def data_loading():

    try:

        summary = get_data_summary()

        return render_template(
            "index.html",
            active="data-loading",
            summary=summary
        )

    except Exception as e:

        return render_template(
            "index.html",
            active="data-loading",
            error=str(e)
        )


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

@app.route("/eda")
def eda():

    try:

        generate_eda_charts()

        eda_summary = get_eda_summary()

        return render_template(
            "eda.html",
            active="eda",
            eda=eda_summary
        )

    except Exception as e:

        return render_template(
            "eda.html",
            active="eda",
            error=str(e)
        )


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing():

    try:

        result = get_preprocessing_summary()

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            result=result
        )

    except Exception as e:

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            error=str(e)
        )


# ============================================================
# LINEAR REGRESSION
# ============================================================

@app.route("/linear-regression")
def linear_regression():

    try:

        result = get_linear_regression()

        return render_template(
            "linear_regression.html",
            active="linear-regression",
            result=result
        )

    except Exception as e:

        return render_template(
            "linear_regression.html",
            active="linear-regression",
            error=str(e)
        )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

@app.route("/logistic-regression")
def logistic_regression():

    try:

        result = get_logistic_regression()

        return render_template(
            "logistic_regression.html",
            active="logistic-regression",
            result=result
        )

    except Exception as e:

        return render_template(
            "logistic_regression.html",
            active="logistic-regression",
            error=str(e)
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )