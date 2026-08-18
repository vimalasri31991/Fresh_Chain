import pandas as pd

from sklearn.model_selection import train_test_split

from ml.data_loader import load_data


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()


print("Dataset shape:")

print(
    df.shape
)


# ============================================================
# ORIGINAL CLASS VALUES
# ============================================================

print("\nOriginal Class values:")

print(
    df["Class"].value_counts()
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["Class"]
)


# ============================================================
# NORMALIZE CLASS LABELS
# ============================================================

train_df["Class_Normalized"] = (
    train_df["Class"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "bad": "Bad",
        "good": "Good"
    })
)


test_df["Class_Normalized"] = (
    test_df["Class"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "bad": "Bad",
        "good": "Good"
    })
)


# ============================================================
# CLASS ENCODING
# ============================================================

class_mapping = {
    "Bad": 0,
    "Good": 1
}


train_df["Class_encoded"] = (
    train_df["Class_Normalized"]
    .map(class_mapping)
)


test_df["Class_encoded"] = (
    test_df["Class_Normalized"]
    .map(class_mapping)
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nTraining data:")

print(
    train_df[
        [
            "Class",
            "Class_Normalized",
            "Class_encoded"
        ]
    ].head(10)
)


print("\nTesting data:")

print(
    test_df[
        [
            "Class",
            "Class_Normalized",
            "Class_encoded"
        ]
    ].head(10)
)


print("\nClass mapping:")

print(
    class_mapping
)