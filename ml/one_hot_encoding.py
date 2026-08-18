import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ml.data_loader import load_data


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()

print("Dataset shape:")
print(df.shape)


print("\nColumns:")
print(
    df.columns.tolist()
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


print("\nTraining data shape:")
print(train_df.shape)

print("\nTesting data shape:")
print(test_df.shape)


# ============================================================
# NOMINAL COLUMN
# ============================================================

nominal_cols = [
    "Fruit"
]


# ============================================================
# ONE HOT ENCODER
# ============================================================

ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# ============================================================
# FIT ON TRAINING DATA
# ============================================================

train_ohe = ohe.fit_transform(
    train_df[nominal_cols]
)


# ============================================================
# TRANSFORM TESTING DATA
# ============================================================

test_ohe = ohe.transform(
    test_df[nominal_cols]
)


# ============================================================
# GENERATED COLUMNS
# ============================================================

ohe_cols = (
    ohe.get_feature_names_out(
        nominal_cols
    )
)


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

train_ohe_df = pd.DataFrame(
    train_ohe,
    columns=ohe_cols,
    index=train_df.index
)


test_ohe_df = pd.DataFrame(
    test_ohe,
    columns=ohe_cols,
    index=test_df.index
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\nNumber of original categorical columns:",
    len(nominal_cols)
)


print(
    "Number of generated encoded columns:",
    len(ohe_cols)
)


print("\nGenerated encoded columns:")

for col in ohe_cols:
    print(col)


print("\nFirst 5 rows of encoded training data:")

print(
    train_ohe_df.head()
)


print("\nFirst 5 rows of encoded testing data:")

print(
    test_ohe_df.head()
)