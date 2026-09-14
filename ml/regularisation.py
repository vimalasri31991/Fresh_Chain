from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from ml.model_data import get_model_data


# ============================================================
# TRAIN AND EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    return {

        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "confusion_matrix":
            confusion_matrix(
                y_test,
                predictions
            )

    }


# ============================================================
# REGULARISATION
# ============================================================

def get_regularisation():

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
    # STANDARD SCALING
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # ========================================================
    # L1 REGULARISATION
    # ========================================================

    l1_model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=1.0,
        max_iter=1000
    )


    l1_result = evaluate_model(
        l1_model,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )


    # ========================================================
    # L2 REGULARISATION
    # ========================================================

    l2_model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=1.0,
        max_iter=1000
    )


    l2_result = evaluate_model(
        l2_model,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "train_rows":
            len(X_train),

        "test_rows":
            len(X_test),

        "l1":
            l1_result,

        "l2":
            l2_result,

        "features":
            list(X.columns)

    }