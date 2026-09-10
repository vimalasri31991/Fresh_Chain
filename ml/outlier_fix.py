from ml.data_loader import load_data


NUMERIC_COLUMNS = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]


def fix_outliers():

    df = load_data().copy()

    summary = []


    for column in NUMERIC_COLUMNS:

        Q1 = df[
            column
        ].quantile(0.25)

        Q3 = df[
            column
        ].quantile(0.75)

        IQR = Q3 - Q1

        lower = (
            Q1 - 1.5 * IQR
        )

        upper = (
            Q3 + 1.5 * IQR
        )


        count = (

            (

                df[column] < lower

            )

            |

            (

                df[column] > upper

            )

        ).sum()


        df[column] = (
            df[column]
            .clip(
                lower=lower,
                upper=upper
            )
        )


        summary.append({

            "Feature":
                column,

            "Outliers":
                int(count),

            "Lower":
                float(lower),

            "Upper":
                float(upper)
        })


    return {

        "data": df,

        "summary": summary
    }