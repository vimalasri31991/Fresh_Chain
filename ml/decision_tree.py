from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

from ml.model_data import get_model_data


# ============================================================
# DECISION TREE
# ============================================================

def get_decision_tree():

    X, y = get_model_data()


    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    )


    # ========================================================
    # TRAIN
    # ========================================================

    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    train_prediction = model.predict(
        X_train
    )

    test_prediction = model.predict(
        X_test
    )


    # ========================================================
    # METRICS
    # ========================================================

    return {

        "train_rows":
            len(X_train),

        "test_rows":
            len(X_test),

        "train_accuracy":
            accuracy_score(
                y_train,
                train_prediction
            ),

        "test_accuracy":
            accuracy_score(
                y_test,
                test_prediction
            ),

        "precision":
            precision_score(
                y_test,
                test_prediction,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                test_prediction,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_test,
                test_prediction,
                zero_division=0
            ),

        "confusion_matrix":
            confusion_matrix(
                y_test,
                test_prediction
            ),

        "tree_depth":
            model.get_depth(),

        "leaf_count":
            model.get_n_leaves(),

        "features":
            list(X.columns)

    }