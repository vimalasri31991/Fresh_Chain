"""
Code shown on the model pages.  Every snippet mirrors the real code in
the matching ml/*.py file, so what the page shows is what actually ran.
"""

_SPLIT = """# 80 / 20 split, same random_state for every model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)"""


def _model(title, tag, body):
    return {"title": title, "tag": tag, "code": body.strip("\n")}


SNIPPETS = {

    # ---------------- RANDOM FOREST ----------------
    "random-forest": [
        _model("Split the data", "Setup", _SPLIT),
        _model("Random Forest · Before tuning", "Before", """
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)"""),
        _model("Random Forest · After tuning", "After", """
model = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)"""),
    ],

    # ---------------- ADABOOST ----------------
    "ada-boost": [
        _model("Split the data", "Setup", _SPLIT),
        _model("AdaBoost · Before tuning", "Before", """
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)
model.fit(X_train, y_train)"""),
        _model("AdaBoost · After tuning", "After", """
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)"""),
    ],

    # ---------------- GRADIENT BOOST ----------------
    "gradient-boost": [
        _model("Split the data", "Setup", _SPLIT),
        _model("Gradient Boosting · Before tuning", "Before", """
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)"""),
        _model("Gradient Boosting · After tuning", "After", """
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=2,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)"""),
    ],

    # ---------------- XGBOOST ----------------
    "xg-boost": [
        _model("Split the data", "Setup", _SPLIT),
        _model("XGBoost · Before tuning", "Before", """
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)"""),
        _model("XGBoost · After tuning", "After", """
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)"""),
    ],

    # ---------------- LIGHTGBM ----------------
    "lightgbm": [
        _model("Split the data", "Setup", _SPLIT),
        _model("LightGBM · Before tuning", "Before", """
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    verbosity=-1
)
model.fit(X_train, y_train)"""),
        _model("LightGBM · After tuning", "After", """
model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=20,
    max_depth=5,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)
model.fit(X_train, y_train)"""),
    ],

    # ---------------- PRUNING (4 sections) ----------------
    "pruning-fully-grown": [
        _model("Fully Grown Tree", "No pruning", """
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print(model.get_depth(), model.get_n_leaves())"""),
    ],

    "pruning-pre-pruning": [
        _model("Pre-Pruning (stop early)", "Limits while growing", """
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)"""),
    ],

    "pruning-post-pruning": [
        _model("Post-Pruning (reduced-error)", "Prune after growing", """
# 1) grow a FULL tree on 75% of the training data
X_grow, X_val, y_grow, y_val = train_test_split(
    X_train, y_train, test_size=0.25,
    random_state=42, stratify=y_train
)
grown = DecisionTreeClassifier(random_state=42).fit(X_grow, y_grow)

# 2) bottom-up: replace a sub-tree by a leaf when that does
#    not increase the error on the validation part
def prune(node, idx):
    leaf_err = (y_val[idx] != node_class[node]).sum()
    if left[node] == -1:
        return leaf_err
    go_left = X_val[idx, feature[node]] <= threshold[node]
    sub_err = prune(left[node], idx[go_left]) + \\
              prune(right[node], idx[~go_left])
    if leaf_err <= sub_err:
        pruned.add(node)          # collapse this branch
        return leaf_err
    return sub_err"""),
    ],

    "pruning-cost-complexity": [
        _model("Cost-Complexity Pruning (ccp_alpha)", "Alpha chosen by CV", """
path = DecisionTreeClassifier(random_state=42) \\
           .cost_complexity_pruning_path(X_train, y_train)

# 5-fold CV accuracy for every candidate alpha (train data only)
scores = [
    cross_val_score(
        DecisionTreeClassifier(random_state=42, ccp_alpha=a),
        X_train, y_train, cv=5
    ).mean()
    for a in path.ccp_alphas
]

# one-standard-error rule: the simplest tree that is still
# within 1 SE of the best cross-validated score
best_alpha = max(a for a, s in zip(path.ccp_alphas, scores)
                 if s >= max(scores) - se)

model = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
model.fit(X_train, y_train)"""),
    ],
}