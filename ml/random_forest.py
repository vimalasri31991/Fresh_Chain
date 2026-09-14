from sklearn.ensemble import RandomForestClassifier

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
# RANDOM FOREST
# ============================================================

def get_random_forest():

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

    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        random_state=42,
        n_jobs=-1
    )


    # ========================================================
    # TRAIN
    # ========================================================

    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    train_prediction = model.predict(
        X_train
    )

    test_prediction = model.predict(
        X_test
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance = {

        feature:
            round(
                float(importance),
                4
            )

        for feature, importance

        in zip(
            X.columns,
            model.feature_importances_
        )

    }


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

        "feature_importance":
            feature_importance

    }