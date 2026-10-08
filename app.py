import matplotlib

matplotlib.use("Agg")   # charts are saved to files - never open a GUI window

import threading

from flask import Flask, render_template, request

from ml.data_loader import get_data_summary
from ml.eda import get_eda_summary, generate_eda_charts
from ml.preprocessing import get_preprocessing_summary

from ml.linear_regression import get_linear_regression
from ml.logistic_regression import get_logistic_regression
from ml.regularisation import get_regularisation

from ml.decision_tree import get_decision_tree
from ml.random_forest import get_random_forest
from ml.cost_complexity import get_cost_complexity
from ml.ada_boost import get_ada_boost
from ml.gradient_boost import get_gradient_boost

# New Tree & Ensemble models
from ml.xg_boost import get_xg_boost
from ml.lightgbm import get_lightgbm
from ml.pruning import get_pruning

# Clustering
from ml.k_calculation import calculate_k
from ml.kmeanfinal import perform_kmeans
from ml.kmeansperformance import evaluate_kmeans

from ml.hierarchical_clustering import get_hierarchical_clustering
from ml.dbscan_clustering import get_dbscan_clustering

from ml.model_comparison import (
    get_result,
    build_comparison,
    ENDPOINT_TO_PAGE,
    PAGE_ORDER,
)
from ml.code_snippets import SNIPPETS


app = Flask(__name__)


# ============================================================
# TEMPLATE HELPERS
#   model_comparison()  -> footer table (before vs after) of the page
#   code_snippets       -> code cards shown on the model pages
# ============================================================

@app.context_processor
def inject_helpers():

    def model_comparison():

        page = ENDPOINT_TO_PAGE.get(request.endpoint)

        return build_comparison(page) if page else None

    return {
        "model_comparison": model_comparison,
        "code_snippets": SNIPPETS,
    }


# ============================================================
# WARM-UP: run every model once in the background so pages and
# comparison tables open instantly
# ============================================================

def _warm_up():

    try:
        generate_eda_charts()
    except Exception:
        pass

    for page in PAGE_ORDER:
        try:
            get_result(page)
        except Exception:
            pass


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
# DATA SET
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

        result = get_result("linear-regression")

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

        result = get_result("logistic-regression")

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
# REGULARISATION
# ============================================================

@app.route("/regularisation")
def regularisation():

    try:

        result = get_result("regularisation")

        return render_template(
            "regularisation.html",
            active="regularisation",
            result=result
        )

    except Exception as e:

        return render_template(
            "regularisation.html",
            active="regularisation",
            error=str(e)
        )


# ============================================================
# DECISION TREE
# ============================================================

@app.route("/decision-tree")
def decision_tree():

    try:

        result = get_result("decision-tree")

        return render_template(
            "decision_tree.html",
            active="decision-tree",
            result=result
        )

    except Exception as e:

        return render_template(
            "decision_tree.html",
            active="decision-tree",
            error=str(e)
        )


# ============================================================
# RANDOM FOREST
# ============================================================

@app.route("/random-forest")
def random_forest():

    try:

        result = get_result("random-forest")

        return render_template(
            "random_forest.html",
            active="random-forest",
            result=result
        )

    except Exception as e:

        return render_template(
            "random_forest.html",
            active="random-forest",
            error=str(e)
        )


# ============================================================
# COST COMPLEXITY
# ============================================================

@app.route("/cost-complexity")
def cost_complexity():

    try:

        result = get_result("cost-complexity")

        return render_template(
            "cost_complexity.html",
            active="cost-complexity",
            result=result
        )

    except Exception as e:

        return render_template(
            "cost_complexity.html",
            active="cost-complexity",
            error=str(e)
        )


# ============================================================
# ADABOOST
# ============================================================

@app.route("/ada-boost")
def ada_boost():

    try:

        result = get_result("ada-boost")

        return render_template(
            "ada_boost.html",
            active="ada-boost",
            result=result
        )

    except Exception as e:

        return render_template(
            "ada_boost.html",
            active="ada-boost",
            error=str(e)
        )


# ============================================================
# GRADIENT BOOST
# ============================================================

@app.route("/gradient-boost")
def gradient_boost():

    try:

        result = get_result("gradient-boost")

        return render_template(
            "gradient_boost.html",
            active="gradient-boost",
            result=result
        )

    except Exception as e:

        return render_template(
            "gradient_boost.html",
            active="gradient-boost",
            error=str(e)
        )


# ============================================================
# XGBOOST
# ============================================================

@app.route("/xg-boost")
def xg_boost():

    try:

        result = get_result("xg-boost")

        if result.get("available") is False:

            return render_template(
                "xg_boost.html",
                active="xg-boost",
                result=None,
                error=result.get(
                    "error",
                    "XGBoost is not available."
                )
            )

        return render_template(
            "xg_boost.html",
            active="xg-boost",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "xg_boost.html",
            active="xg-boost",
            result=None,
            error=str(e)
        )


# ============================================================
# LIGHTGBM
# ============================================================

@app.route("/lightgbm")
def lightgbm():

    try:

        result = get_result("lightgbm")

        if result.get("available") is False:

            return render_template(
                "lightgbm.html",
                active="lightgbm",
                result=None,
                error=result.get(
                    "error",
                    "LightGBM is not available."
                )
            )

        return render_template(
            "lightgbm.html",
            active="lightgbm",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "lightgbm.html",
            active="lightgbm",
            result=None,
            error=str(e)
        )


# ============================================================
# PRUNING
# ============================================================

@app.route("/pruning")
def pruning():

    try:

        result = get_result("pruning")

        return render_template(
            "pruning.html",
            active="pruning",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "pruning.html",
            active="pruning",
            result=None,
            error=str(e)
        )


# ============================================================
# K-MEANS
# ============================================================

@app.route("/kmeans", methods=["GET", "POST"])
def kmeans():

    k_results = None
    performance = None
    cluster_counts = None
    selected_k = None
    selected_method = None
    error = None

    try:

        k_results = get_result("kmeans")

        if request.method == "POST":

            selected_method = request.form.get(
                "method",
                ""
            ).strip()

            k_text = request.form.get(
                "k",
                ""
            ).strip()

            if selected_method not in [
                "elbow",
                "silhouette"
            ]:

                raise ValueError(
                    "Please select Elbow Method or "
                    "Silhouette Method."
                )

            if not k_text:

                raise ValueError(
                    "Please select a K value."
                )

            selected_k = int(k_text)

            if selected_k < 2 or selected_k > 10:

                raise ValueError(
                    "K value must be between 2 and 10."
                )

            perform_kmeans(
                selected_k
            )

            performance = evaluate_kmeans(
                selected_k
            )

            cluster_counts = performance.get(
                "clusters"
            )

    except Exception as e:

        error = str(e)

    return render_template(
        "kmeans.html",
        active="kmeans",
        k_results=k_results,
        performance=performance,
        cluster_counts=cluster_counts,
        selected_k=selected_k,
        selected_method=selected_method,
        error=error
    )


# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================

@app.route("/hierarchical")
def hierarchical():

    try:

        result = get_result("hierarchical")

        return render_template(
            "hierarchical.html",
            active="hierarchical",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "hierarchical.html",
            active="hierarchical",
            result=None,
            error=str(e)
        )


# ============================================================
# DBSCAN
# ============================================================

@app.route("/dbscan")
def dbscan():

    try:

        result = get_result("dbscan")

        return render_template(
            "dbscan.html",
            active="dbscan",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "dbscan.html",
            active="dbscan",
            result=None,
            error=str(e)
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    import os

    # start the warm-up once (not in the debug reloader's parent process)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        threading.Thread(target=_warm_up, daemon=True).start()

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )