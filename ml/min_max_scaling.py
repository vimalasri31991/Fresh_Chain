import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from ml.data_loader import load_data


NUMERIC_COLUMNS = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]


def min_max_scale():

    df = load_data()

    data = df[
        NUMERIC_COLUMNS
    ].dropna()

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        data
    )

    result = pd.DataFrame(
        scaled,
        columns=NUMERIC_COLUMNS
    )

    return result