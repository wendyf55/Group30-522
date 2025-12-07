import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_ORIGIN_PATH = "../data/raw/abalone.data"
DEFAULT_OUTPUT_DIR = "../data/processed"

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
    Reads in the Abalone dataset from a filepath, adds in correct column headers, then cleans and splits the data and sends it to the destination.
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


@click.command()
@click.option(
    "--origin_path",
    type=str,
    help="Path to directory where raw data is located",
    default=DEFAULT_ORIGIN_PATH,
    show_default=True,
)
@click.option(
    "--output_dir",
    type=str,
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Path to directory where clean and split data will be written into",
)
def main(origin_path, output_dir):
    """Reads in Abalone data from directory, cleans, and splits it."""
    read_and_split(origin_path, output_dir)


if __name__ == "__main__":
    main()
