import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from ml.data_loader import load_data


def create_encoder():

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def one_hot_encode():

    df = load_data()

    data = df[
        ["Fruit"]
    ].fillna("Unknown")


    encoder = create_encoder()

    encoded = encoder.fit_transform(
        data
    )


    columns = (
        encoder
        .get_feature_names_out(
            ["Fruit"]
        )
    )


    result = pd.DataFrame(
        encoded,
        columns=columns
    )


    return result