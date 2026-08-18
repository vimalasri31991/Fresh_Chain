import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.data_loader import load_data


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()

print("Original dataset shape:")
print(df.shape)


# ============================================================
# NUMERICAL COLUMNS
# ============================================================

numeric_cols = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]


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
# STANDARD SCALER
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    train_df[numeric_cols]
)


X_test_scaled = scaler.transform(
    test_df[numeric_cols]
)


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=numeric_cols,
    index=train_df.index
)


X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=numeric_cols,
    index=test_df.index
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nTraining Data Before Standard Scaling:")

print(
    train_df[numeric_cols].head()
)


print("\nStandard Scaled Training Data:")

print(
    X_train_scaled_df.head()
)


print("\nStandard Scaled Testing Data:")

print(
    X_test_scaled_df.head()
)


# ============================================================
# CHECK MEAN
# ============================================================

print("\nMean after Standard Scaling:")

print(
    X_train_scaled_df.mean().round(3)
)


# ============================================================
# CHECK STANDARD DEVIATION
# ============================================================

print("\nStandard deviation after Standard Scaling:")

print(
    X_train_scaled_df.std().round(3)
)