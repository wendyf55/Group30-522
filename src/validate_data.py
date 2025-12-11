import pandas as pd
import pandera.pandas as pa


def validate_data(train_df):
    """
    This schema checks column types and that there are no NULL values in the feature columns.

    Parameters
    ----------
    train_df : pd.DataFrame
        Dataframe to validate.

    Raises
    ------
    pandera.errors.SchemaError
        If the dataframe contains incorrect data types or NULL values.

    Examples
    --------
    >>> check_data_type_and_null(train_df)
    """
    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if train_df.empty:
        raise ValueError("Dataframe must contain observations.")

    schema = pa.DataFrameSchema({
        "Sex": pa.Column(str, nullable=False),
        "Length": pa.Column(float, nullable=False),
        "Diameter": pa.Column(float, nullable=False),
        "Height": pa.Column(float, nullable=False),
        "Whole_weight": pa.Column(float, nullable=False),
        "Shucked_weight": pa.Column(float, nullable=False),
        "Viscera_weight": pa.Column(float, nullable=False),
        "Shell_weight": pa.Column(float, nullable=False),
        "Rings": pa.Column(int)
    }
    )

    schema.validate(train_df, lazy=True)
