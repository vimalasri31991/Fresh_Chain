"""
Safety net for the classifier pages (Decision Tree, Random Forest,
AdaBoost, Gradient Boost).

If a model function returns a result that is missing the standard
before/after keys the pages need, the same models are evaluated here
with the exact settings shown on the pages, so the pages never crash.
A one-line message is printed in the console when this happens.
"""

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml.model_data import get_model_data


REQUIRED = [
    "before_accuracy", "after_accuracy",
    "before_precision", "after_precision",
    "before_recall", "after_recall",
    "before_f1", "after_f1",
    "train_rows", "test_rows",
]


def _ada(**kw):
    try:
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=kw.pop("depth"), random_state=42),
            random_state=42, **kw
        )
    except TypeError:                      # old scikit-learn
        return AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=kw.pop("depth"), random_state=42),
            random_state=42, **kw
        )


MODELS = {
    "decision-tree": (
        lambda: DecisionTreeClassifier(criterion="gini", random_state=42),
        lambda: DecisionTreeClassifier(criterion="gini", max_depth=5,
                                       min_samples_split=10,
                                       min_samples_leaf=5, random_state=42),
        {},
    ),
    "random-forest": (
        lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        lambda: RandomForestClassifier(n_estimators=200, max_depth=10,
                                       min_samples_split=5, min_samples_leaf=2,
                                       max_features="sqrt", random_state=42, n_jobs=-1),
        {"n_estimators": 200, "max_depth": 10},
    ),
    "ada-boost": (
        lambda: _ada(depth=1, n_estimators=100, learning_rate=0.5),
        lambda: _ada(depth=2, n_estimators=200, learning_rate=0.1),
        {"n_estimators": 200, "learning_rate": 0.1},
    ),
    "gradient-boost": (
        lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                           max_depth=3, random_state=42),
        lambda: GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                           max_depth=2, min_samples_split=5,
                                           min_samples_leaf=2, subsample=0.8,
                                           random_state=42),
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2},
    ),
}


def _score(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return model, {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "cm": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    }


def build_result(page):

    make_before, make_after, extra = MODELS[page]

    X, y = get_model_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    _, before = _score(make_before(), X_train, X_test, y_train, y_test)
    after_model, after = _score(make_after(), X_train, X_test, y_train, y_test)

    result = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": list(X.columns),
        "confusion_matrix": after["cm"],
    }

    for key in ("accuracy", "precision", "recall", "f1"):
        result[f"before_{key}"] = before[key]
        result[f"after_{key}"] = after[key]

    result.update(extra)

    if page == "decision-tree":
        before_model = make_before().fit(X_train, y_train)
        result["before_depth"] = int(before_model.get_depth())
        result["after_depth"] = int(after_model.get_depth())
        result["before_leaves"] = int(before_model.get_n_leaves())
        result["after_leaves"] = int(after_model.get_n_leaves())

    if hasattr(after_model, "feature_importances_"):
        result["feature_importance"] = {
            name: round(float(v), 4)
            for name, v in zip(X.columns, after_model.feature_importances_)
        }

    return result


def ensure_classifier_result(page, result):

    if page not in MODELS:
        return result

    if isinstance(result, dict) and all(k in result for k in REQUIRED):
        return result

    found = sorted(result.keys()) if isinstance(result, dict) else type(result).__name__
    print(f"[FreshChain] '{page}' result is missing standard keys "
          f"(found: {found}); using built-in evaluation instead.")

    return build_result(page)