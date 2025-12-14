import os
import pandas as pd
from sklearn.model_selection import train_test_split

COLUMN_NAMES = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
    "Rings",
]


def read_and_split(origin_path, output_dir):
    """
    Load the Abalone dataset, clean it, split it into training and test
    sets, and write the resulting datasets to CSV files.

    The function reads the raw Abalone data (without headers), assigns
    column names, removes rows containing missing values, separates features and
    target, performs an 80/20 train–test split, and saves the resulting datasets
    to the specified output directory.

    Parameters
    ----------
    origin_path : str
        File path to the raw Abalone dataset CSV.

    output_dir : str
        Directory where the processed training and test CSV files will be saved.
        The directory is created if it does not already exist.

    Returns
    -------
    abalone_test.csv
    abalone_train.csv

    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(origin_path, header=None, names=COLUMN_NAMES)

    df = df.dropna()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=522
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = os.path.join(output_dir, "abalone_train.csv")
    test_path = os.path.join(output_dir, "abalone_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
