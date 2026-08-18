from flask import Flask, render_template

from ml.data_loader import get_data_summary

from ml.eda import (
    get_eda_summary,
    generate_eda_charts
)

from ml.preprocessing import (
    get_preprocessing_summary
)


app = Flask(__name__)


# ============================================================
# HOME / DASHBOARD
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none"
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
# EDA
# ============================================================

@app.route("/eda")
def eda():

    try:

        eda_summary = get_eda_summary()

        generate_eda_charts()

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

        preprocessing_summary = (
            get_preprocessing_summary()
        )

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            preprocessing=preprocessing_summary
        )

    except Exception as e:

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            error=str(e)
        )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )