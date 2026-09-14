from sklearn.ensemble import AdaBoostClassifier

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
# ADABOOST
# ============================================================

def get_ada_boost():

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
    # BASE ESTIMATOR
    # ========================================================

    base_tree = DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    )


    # ========================================================
    # MODEL
    # ========================================================

    try:

        model = AdaBoostClassifier(
            estimator=base_tree,
            n_estimators=100,
            learning_rate=0.5,
            random_state=42
        )

    except TypeError:

        model = AdaBoostClassifier(
            base_estimator=base_tree,
            n_estimators=100,
            learning_rate=0.5,
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
    # RETURN
    # ========================================================

    return {

        "train_rows":
            len(X_train),

        "test_rows":
            len(X_test),

        "n_estimators":
            model.n_estimators,

        "learning_rate":
            model.learning_rate,

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
            )

    }