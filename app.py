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

from ml.k_calculation import calculate_k
from ml.kmeanfinal import perform_kmeans
from ml.kmeansperformance import evaluate_kmeans
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template(
        "index.html",
        active="dashboard"
    )


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


@app.route("/regularisation")
def regularisation():
    try:
        result = get_regularisation()

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


@app.route("/decision-tree")
def decision_tree():
    try:
        result = get_decision_tree()

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


@app.route("/random-forest")
def random_forest():
    try:
        result = get_random_forest()

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


@app.route("/cost-complexity")
def cost_complexity():
    try:
        result = get_cost_complexity()

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


@app.route("/ada-boost")
def ada_boost():
    try:
        result = get_ada_boost()

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


@app.route("/gradient-boost")
def gradient_boost():
    try:
        result = get_gradient_boost()

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


@app.route("/kmeans", methods=["GET", "POST"])
def kmeans():

    k_results = None
    performance = None
    cluster_counts = None
    selected_k = None
    selected_method = None
    error = None

    try:

        k_results = calculate_k()

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
                    "Please select Elbow Method or Silhouette Method."
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

            perform_kmeans(selected_k)

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


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )