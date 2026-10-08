"""
Model registry + cumulative comparison table.

* get_result(key)       -> runs a model ONCE and caches the result
* build_comparison(key) -> rows for every model finished up to `key`
                           (used by the footer of every model page)
"""

from threading import Lock

from ml.model_utils import as_fraction, normalise_importance
from ml.model_fallback import ensure_classifier_result


# ============================================================
# PIPELINE ORDER  (same order as the sidebar)
# ============================================================

PAGE_ORDER = [
    "linear-regression",
    "logistic-regression",
    "regularisation",
    "decision-tree",
    "random-forest",
    "ada-boost",
    "gradient-boost",
    "xg-boost",
    "lightgbm",
    "pruning",
    "cost-complexity",
    "kmeans",
    "hierarchical",
    "dbscan",
]

PAGE_TITLES = {
    "linear-regression": "Linear Regression",
    "logistic-regression": "Logistic Regression",
    "regularisation": "Regularisation",
    "decision-tree": "Decision Tree",
    "random-forest": "Random Forest",
    "ada-boost": "AdaBoost",
    "gradient-boost": "Gradient Boost",
    "xg-boost": "XGBoost",
    "lightgbm": "LightGBM",
    "pruning": "Pruning",
    "cost-complexity": "Cost-Complexity",
    "kmeans": "K-Means",
    "hierarchical": "Hierarchical",
    "dbscan": "DBSCAN",
}

# Flask endpoint name -> page key
ENDPOINT_TO_PAGE = {
    "linear_regression": "linear-regression",
    "logistic_regression": "logistic-regression",
    "regularisation": "regularisation",
    "decision_tree": "decision-tree",
    "random_forest": "random-forest",
    "ada_boost": "ada-boost",
    "gradient_boost": "gradient-boost",
    "xg_boost": "xg-boost",
    "lightgbm": "lightgbm",
    "pruning": "pruning",
    "cost_complexity": "cost-complexity",
    "kmeans": "kmeans",
    "hierarchical": "hierarchical",
    "dbscan": "dbscan",
}


# ============================================================
# LOADERS (imported lazily so a missing optional library such as
# XGBoost / LightGBM never breaks the other pages)
# ============================================================

def _loader(page):

    if page == "linear-regression":
        from ml.linear_regression import get_linear_regression
        return get_linear_regression

    if page == "logistic-regression":
        from ml.logistic_regression import get_logistic_regression
        return get_logistic_regression

    if page == "regularisation":
        from ml.regularisation import get_regularisation
        return get_regularisation

    if page == "decision-tree":
        from ml.decision_tree import get_decision_tree
        return get_decision_tree

    if page == "random-forest":
        from ml.random_forest import get_random_forest
        return get_random_forest

    if page == "ada-boost":
        from ml.ada_boost import get_ada_boost
        return get_ada_boost

    if page == "gradient-boost":
        from ml.gradient_boost import get_gradient_boost
        return get_gradient_boost

    if page == "xg-boost":
        from ml.xg_boost import get_xg_boost
        return lambda: _normalise_boost(get_xg_boost())

    if page == "lightgbm":
        from ml.lightgbm import get_lightgbm
        return lambda: _normalise_boost(get_lightgbm())

    if page == "pruning":
        from ml.pruning import get_pruning
        return get_pruning

    if page == "cost-complexity":
        from ml.cost_complexity import get_cost_complexity
        return get_cost_complexity

    if page == "kmeans":
        from ml.k_calculation import calculate_k
        return calculate_k

    if page == "hierarchical":
        from ml.hierarchical_clustering import get_hierarchical_clustering
        return get_hierarchical_clustering

    if page == "dbscan":
        from ml.dbscan_clustering import get_dbscan_clustering
        return get_dbscan_clustering

    raise KeyError(page)


def _normalise_boost(result):
    """XGBoost / LightGBM modules store 0-100 values; the page template
    expects 0-1 like every other classifier, and importances as shares."""

    if not isinstance(result, dict) or result.get("available") is False:
        return result

    for key in list(result):
        if key.startswith(("before_", "after_")):
            result[key] = as_fraction(result[key])

    if isinstance(result.get("feature_importance"), dict):
        result["feature_importance"] = normalise_importance(
            result["feature_importance"]
        )

    return result


# ============================================================
# CACHE
# ============================================================

_CACHE = {}
_LOCK = Lock()


def get_result(page):
    """Run a model once per server session and reuse the result."""

    with _LOCK:
        if page not in _CACHE:
            _CACHE[page] = ensure_classifier_result(page, _loader(page)())
        return _CACHE[page]


def clear_cache():
    with _LOCK:
        _CACHE.clear()


# ============================================================
# ROWS
# ============================================================

def _pct(value):
    return None if value is None else round(float(value), 2)


def _rows_for(page):
    """Return comparison rows (dicts) contributed by one page."""

    r = get_result(page)

    if isinstance(r, dict) and r.get("available") is False:
        return []

    def row(label, family, metric, before, after):
        return {
            "page": page,
            "label": label,
            "family": family,
            "metric": metric,
            "before": _pct(before),
            "after": _pct(after),
        }

    if page == "linear-regression":
        return [row("Linear Regression", "Regression", "R² score",
                    None, r["r2"] * 100)]

    if page == "logistic-regression":
        unscaled = next(
            (c["test_accuracy"] for c in r["comparison"] if c["name"] == "Unscaled"),
            None,
        )
        return [row("Logistic Regression", "Regression", "Accuracy",
                    unscaled, r["best_accuracy"])]

    if page == "regularisation":
        lin, log = r["linear"], r["logistic"]
        return [
            row("Linear · L1 / L2", "Regression", "R² score",
                lin["baseline"]["r2"] * 100,
                max(lin["l1"]["r2"], lin["l2"]["r2"]) * 100),
            row("Logistic · L1 / L2", "Regression", "Accuracy",
                log["baseline"]["accuracy"] * 100,
                max(log["l1"]["accuracy"], log["l2"]["accuracy"]) * 100),
        ]

    if page in ("decision-tree", "random-forest", "ada-boost",
                "gradient-boost", "xg-boost", "lightgbm"):
        family = "Tree" if page == "decision-tree" else "Ensemble"
        return [row(PAGE_TITLES[page], family, "Accuracy",
                    r["before_accuracy"] * 100, r["after_accuracy"] * 100)]

    if page == "pruning":
        base = r["fully_grown"]["accuracy"]
        return [
            row("Pruning · Fully Grown", "Tree", "Accuracy", None, base),
            row("Pruning · Pre-Pruning", "Tree", "Accuracy", base,
                r["pre_pruning"]["accuracy"]),
            row("Pruning · Post-Pruning", "Tree", "Accuracy", base,
                r["post_pruning"]["accuracy"]),
            row("Pruning · Cost-Complexity", "Tree", "Accuracy", base,
                r["cost_complexity"]["accuracy"]),
        ]

    if page == "cost-complexity":
        return [row("Cost-Complexity (validation)", "Tree", "Accuracy",
                    None, r["test_accuracy"])]

    if page == "kmeans":
        scores = r["silhouette_scores"]
        best = max(range(len(scores)), key=lambda i: scores[i])
        return [row(f"K-Means (K={r['k_values'][best]})", "Clustering",
                    "Silhouette", None, scores[best] * 100)]

    if page in ("hierarchical", "dbscan"):
        return [row(PAGE_TITLES[page], "Clustering", "Silhouette",
                    None, r["silhouette_score"] * 100)]

    return []


# ============================================================
# PUBLIC: COMPARISON TABLE
# ============================================================

def build_comparison(current_page):

    if current_page not in PAGE_ORDER:
        return None

    upto = PAGE_ORDER.index(current_page)

    rows = []

    for page in PAGE_ORDER[: upto + 1]:
        try:
            rows.extend(_rows_for(page))
        except Exception:
            # one broken / unavailable model must not hide the others
            continue

    if not rows:
        return None

    for r in rows:
        r["is_current"] = r["page"] == current_page
        r["change"] = (
            None if r["before"] is None or r["after"] is None
            else round(r["after"] - r["before"], 2)
        )

    current_rows = [r for r in rows if r["is_current"]]
    earlier_rows = [r for r in rows if not r["is_current"]]

    # "before this model" vs "after this model": best score of the same
    # kind of model (same metric) so we never compare accuracy with R².
    summary = None

    if current_rows:
        metric = current_rows[0]["metric"]
        this_score = max(r["after"] for r in current_rows)
        earlier = [r["after"] for r in earlier_rows if r["metric"] == metric]

        summary = {
            "metric": metric,
            "this_model": this_score,
            "best_before": max(earlier) if earlier else None,
            "models_done": len({r["page"] for r in rows}),
            "models_total": len(PAGE_ORDER),
        }

        summary["delta"] = (
            None if summary["best_before"] is None
            else round(this_score - summary["best_before"], 2)
        )

    top = max(r["after"] for r in rows) or 1.0

    for r in rows:
        r["bar"] = round(max(r["after"], 0) / max(top, 1) * 100, 1)

    return {
        "current_page": current_page,
        "current_title": PAGE_TITLES[current_page],
        "rows": rows,
        "summary": summary,
    }