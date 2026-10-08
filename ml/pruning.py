from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml.model_data import get_model_data


BASE_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = BASE_DIR / "static" / "charts"


# ============================================================
# METRICS
# ============================================================

def _metrics(y_true, prediction, depth, leaves):

    return {
        "accuracy": round(float(accuracy_score(y_true, prediction)) * 100, 2),
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_true, prediction, zero_division=0)) * 100, 2),
        "f1": round(float(f1_score(y_true, prediction, zero_division=0)) * 100, 2),
        "depth": int(depth),
        "leaves": int(leaves),
    }


def _fit_and_score(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    return _metrics(
        y_test,
        model.predict(X_test),
        model.get_depth(),
        model.get_n_leaves(),
    )


# ============================================================
# POST-PRUNING  (reduced-error pruning)
# ============================================================
# 1. grow a FULL tree on the training part
# 2. walk bottom-up and replace a sub-tree by a leaf whenever that
#    does not increase the error on a separate validation part

def _reduced_error_prune(model, X_val, y_val):

    tree = model.tree_
    left, right = tree.children_left, tree.children_right
    feature, threshold = tree.feature, tree.threshold
    node_class = model.classes_[tree.value[:, 0, :].argmax(axis=1)]

    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    pruned = set()

    def prune(node, idx):

        leaf_errors = int((y_val[idx] != node_class[node]).sum())

        if left[node] == -1:
            return leaf_errors

        goes_left = X_val[idx, feature[node]] <= threshold[node]

        sub_errors = (
            prune(left[node], idx[goes_left])
            + prune(right[node], idx[~goes_left])
        )

        if leaf_errors <= sub_errors:
            pruned.add(node)
            return leaf_errors

        return sub_errors

    prune(0, np.arange(len(y_val)))

    def predict(X):

        X = np.asarray(X)
        out = []

        for row in X:
            node = 0
            while left[node] != -1 and node not in pruned:
                node = left[node] if row[feature[node]] <= threshold[node] else right[node]
            out.append(node_class[node])

        return np.array(out)

    def shape(node=0, depth=0):

        if left[node] == -1 or node in pruned:
            return depth, 1

        d1, l1 = shape(left[node], depth + 1)
        d2, l2 = shape(right[node], depth + 1)

        return max(d1, d2), l1 + l2

    depth, leaves = shape()

    return predict, depth, leaves


# ============================================================
# MAIN
# ============================================================

def get_pruning():

    X, y = get_model_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ---------------- 1. FULLY GROWN ----------------
    fully_grown = _fit_and_score(
        DecisionTreeClassifier(random_state=42),
        X_train, X_test, y_train, y_test
    )
    fully_grown["params"] = {"max_depth": "None", "ccp_alpha": 0}

    # ---------------- 2. PRE-PRUNING ----------------
    pre_params = {
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
    }
    pre_pruning = _fit_and_score(
        DecisionTreeClassifier(random_state=42, **pre_params),
        X_train, X_test, y_train, y_test
    )
    pre_pruning["params"] = pre_params

    # ---------------- 3. POST-PRUNING ----------------
    X_grow, X_val, y_grow, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    grown = DecisionTreeClassifier(random_state=42).fit(X_grow, y_grow)
    full_leaves = int(grown.get_n_leaves())

    predict, depth, leaves = _reduced_error_prune(grown, X_val, y_val)

    post_pruning = _metrics(y_test, predict(X_test), depth, leaves)
    post_pruning["params"] = {
        "grown_leaves": full_leaves,
        "validation_share": "25% of train",
    }

    # ---------------- 4. COST-COMPLEXITY ----------------
    path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(
        X_train, y_train
    )

    alphas = np.unique(np.round(path.ccp_alphas, 8))

    if len(alphas) > 30:
        alphas = alphas[np.linspace(0, len(alphas) - 1, 30).astype(int)]

    scored = []

    for alpha in alphas:
        scores = cross_val_score(
            DecisionTreeClassifier(random_state=42, ccp_alpha=float(alpha)),
            X_train, y_train, cv=5, scoring="accuracy"
        )
        scored.append((float(alpha), float(scores.mean()), float(scores.std() / np.sqrt(5))))

    # one-standard-error rule: simplest tree within 1 SE of the best CV score
    best_mean, best_se = max(((m, se) for _, m, se in scored), key=lambda t: t[0])
    best_alpha = max(a for a, m, _ in scored if m >= best_mean - best_se)

    cost_complexity = _fit_and_score(
        DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha),
        X_train, X_test, y_train, y_test
    )
    cost_complexity["best_alpha"] = round(best_alpha, 6)
    cost_complexity["params"] = {
        "ccp_alpha": round(best_alpha, 6),
        "selected_by": "5-fold CV on train",
    }

    # ---------------- ALPHA CHART ----------------
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    ax.plot([a for a, _, _ in scored], [m * 100 for _, m, _ in scored],
            marker="o", color="#7c58c9")
    ax.axvline(best_alpha, color="#c1477e", linestyle="--",
               label=f"chosen alpha = {best_alpha:.5f}")
    ax.set_xlabel("ccp_alpha")
    ax.set_ylabel("5-fold CV accuracy (%)")
    ax.set_title("Cost-Complexity Pruning Path")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "pruning_alpha_path.png")
    plt.close(fig)

    return {
        "fully_grown": fully_grown,
        "pre_pruning": pre_pruning,
        "post_pruning": post_pruning,
        "cost_complexity": cost_complexity,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "chart": "charts/pruning_alpha_path.png",
    }