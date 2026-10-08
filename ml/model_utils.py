"""Small helpers shared by several ML modules."""

import sklearn
from sklearn.linear_model import LogisticRegression


def _sk_version():
    parts = sklearn.__version__.split(".")[:2]
    return tuple(int("".join(ch for ch in p if ch.isdigit()) or 0) for p in parts)


def make_logistic(penalty, C=1.0, max_iter=5000):
    """
    Build a Logistic Regression with an L1 or L2 penalty.

    scikit-learn 1.8+ deprecated the `penalty` argument in favour of
    `l1_ratio`, so we pick the right spelling for the installed version
    (no deprecation warnings, and the penalty really is applied).
    """
    penalty = penalty.lower()

    if penalty not in ("l1", "l2"):
        raise ValueError("penalty must be 'l1' or 'l2'")

    if _sk_version() >= (1, 8):
        return LogisticRegression(
            l1_ratio=1.0 if penalty == "l1" else 0.0,
            solver="saga",
            C=C,
            max_iter=max_iter,
            random_state=42,
        )

    return LogisticRegression(
        penalty=penalty,
        solver="liblinear",
        C=C,
        max_iter=max_iter,
        random_state=42,
    )


def as_fraction(value):
    """Return a metric on a 0-1 scale (some modules store 0-100)."""
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def normalise_importance(importance):
    """Turn raw importances (gain / split counts) into shares that sum to 1."""
    total = float(sum(importance.values())) or 1.0
    return {k: round(float(v) / total, 4) for k, v in importance.items()}