import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
# BEFORE SCALING
# ============================================================

print("\nTraining Data Before Min-Max Scaling:")

print(
    train_df[numeric_cols].head()
)


print("\nTesting Data Before Min-Max Scaling:")

print(
    test_df[numeric_cols].head()
)


# ============================================================
# MIN-MAX SCALER
# ============================================================

scaler = MinMaxScaler()


train_scaled = scaler.fit_transform(
    train_df[numeric_cols]
)


test_scaled = scaler.transform(
    test_df[numeric_cols]
)


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

train_scaled_df = pd.DataFrame(
    train_scaled,
    columns=numeric_cols,
    index=train_df.index
)


test_scaled_df = pd.DataFrame(
    test_scaled,
    columns=numeric_cols,
    index=test_df.index
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nScaled Training Data:")

print(
    train_scaled_df.head()
)


print("\nScaled Testing Data:")

print(
    test_scaled_df.head()
)


print("\nMinimum values after scaling:")

print(
    train_scaled_df.min()
)


print("\nMaximum values after scaling:")

print(
    train_scaled_df.max()
)