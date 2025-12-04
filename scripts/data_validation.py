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


def save_boxplot(train_df, boxplot_path):
    # Create boxplot with improved formatting
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=train_df, x='Rings', ax=ax, color='steelblue')
    ax.set_xlabel('Number of Rings (Age Proxy)', fontsize=12)
    ax.set_title('Distribution of Abalone Ring Counts', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plt.savefig(boxplot_path)


def save_histogram(train_df, histogram_path):
    # Create histogram with improved formatting
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=train_df, x='Rings', binwidth=1, ax=ax, color='steelblue', edgecolor='white')
    ax.set_xlabel('Number of Rings (Age Proxy)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Abalone Ring Counts', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)

    # Add vertical line for median
    median_rings = train_df['Rings'].median()
    ax.axvline(median_rings, color='red', linestyle='--', linewidth=2, label=f'Median = {median_rings:.0f}')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(histogram_path)
    

def check_target_distribution(train_df):
    from scipy.stats import shapiro
    normal_pvalue = shapiro(train_df.Rings).pvalue
    try:
        assert normal_pvalue > 0.05
    except AssertionError:
        print(f"Target variable is not normal! Shapiro p-value: {normal_pvalue}")


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
@click.option('--boxplot-file',
              type=str,
              default=DEFAULT_BOXPLOT_FILE,
              help='Filename to save boxplot for target distribution')
@click.option('--hist-file',
              type=str,
              default=DEFAULT_HIST_FILE,
              help='Filename to save histogram for target distribution')
def main(train_path, save_dir, boxplot_file, hist_file):
    os.makedirs(save_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)

    check_data_format(train_df)
    check_data_type_and_null(train_df)
    check_missing_target(train_df)
    check_anomalous_continuous(train_df)
    check_anomalous_categorical(train_df)
    check_duplicates(train_df)
    check_empty(train_df)

    save_boxplot(train_df, os.path.join(save_dir, boxplot_file))
    save_histogram(train_df, os.path.join(save_dir, hist_file))
    check_target_distribution(train_df)


if __name__ == "__main__":
    main()
