import click
import os
import pandas as pd
import pandera.pandas as pa
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_TRAIN_PATH = '../data/processed/abalone_train.csv'
DEFAULT_SAVE_DIR = '../results/data_validation'
DEFAULT_BOXPLOT_FILE = 'target_boxplot.png'
DEFAULT_HIST_FILE = 'target_histogram.png'
DEFAULT_CORRELATION_FILE = 'correlation_plot.png'


def check_data_format(train_df):
    assert isinstance(train_df, pd.DataFrame), "Expected 'train_df' to be a Pandas DataFrame"


def check_data_type_and_null(train_df):
    # This schema checks column types, and that there are no NULL values in the feature columns. 
    
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


def check_missing_target(train_df):
    # This checks that there are not more than 5% missing values in the target column

    schema = pa.DataFrameSchema(
        {
            "Rings": pa.Column(
                int,
                pa.Check(lambda s: s.isna().mean() <= 0.05,
                         element_wise=False,
                         error="Too many null values in 'Rings' column."),
                nullable=True
            )
        }
    )

    schema.validate(train_df, lazy=True)


def check_anomalous_continuous(train_df):
    # checking that numeric features are within range; no extreme outliers

    schema = pa.DataFrameSchema(
        {
            "Length": pa.Column(float, pa.Check.between(0, 1)),
            "Diameter": pa.Column(float, pa.Check.between(0, 1)),
            "Height": pa.Column(float, pa.Check.between(0, 1)),
            "Whole_weight": pa.Column(float, pa.Check.between(0, 3)),
            "Shucked_weight": pa.Column(float, pa.Check.between(0, 2)),
            "Viscera_weight": pa.Column(float, pa.Check.between(0, 1)),
            "Shell_weight": pa.Column(float, pa.Check.between(0, 1.10)),
            "Rings": pa.Column(int, pa.Check.between(0, 30))
        }
    )

    schema.validate(train_df, lazy=True)


def check_anomalous_categorical(train_df):
    # Checking that the 'sex' column only has the values M, F, or I

    schema = pa.DataFrameSchema(
        {
            "Sex": pa.Column(str, pa.Check.isin(["M", "F", "I"]), nullable=False)
        }
    )

    schema.validate(train_df, lazy=True)


def check_duplicates(train_df):
    # Checking for duplicates

    schema = pa.DataFrameSchema(
        checks=[
            pa.Check(lambda train_df: ~train_df.duplicated().any(), 
                     error="Duplicate rows found.")
        ]
    )

    schema.validate(train_df, lazy=True)


def check_empty(train_df):
    # Checking for empty observations

    schema = pa.DataFrameSchema(
        checks=[
            pa.Check(lambda train_df: ~train_df.duplicated().any(),
                     error="Duplicate rows found."),
            pa.Check(lambda train_df: ~(train_df.isna().all(axis=1)).any(), 
                     error="Empty rows found.")
        ]
    )

    schema.validate(train_df, lazy=True)


@click.command()
@click.option('--train-path',
              type=str,
              default=DEFAULT_TRAIN_PATH,
              help='Path to train data'
              )
@click.option('--save-dir',
              type=str,
              default=DEFAULT_SAVE_DIR,
              help='Path to directory to save EDA outputs')
def main(train_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)

    check_data_format(train_df)
    check_data_type_and_null(train_df)
    check_missing_target(train_df)
    check_anomalous_continuous(train_df)
    check_anomalous_categorical(train_df)
    check_duplicates(train_df)
    check_empty(train_df)


if __name__ == "__main__":
    main()
