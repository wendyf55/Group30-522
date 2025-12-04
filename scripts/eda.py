import click
import os
import pandas as pd
from ydata_profiling import ProfileReport

DEFAULT_TRAIN_PATH = '../data/processed/abalone_train.csv'
DEFAULT_SAVE_DIR = '../results/eda'
DEFAULT_TABLE_FILE = 'descriptive_stats.csv'
DEFAULT_REPORT_FILE = 'pandas_profiling.html'


def save_descriptive_table(train_df, descriptive_path):
    train_df.describe().round(2).to_csv(descriptive_path)


def save_eda_report(train_df, report_path):
    report = ProfileReport(train_df)
    report.to_file(report_path)


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
@click.option('--table-file',
              type=str,
              default=DEFAULT_TABLE_FILE,
              help='Filename to save descriptive stats including extension')
@click.option('--report-file',
              type=str,
              default=DEFAULT_REPORT_FILE,
              help='Filename to save descriptive stats including extension')
def main(train_path, save_dir, table_file, report_file):
    os.makedirs(save_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)

    save_descriptive_table(train_df, os.path.join(save_dir, table_file))
    save_eda_report(train_df, os.path.join(save_dir, report_file))


if __name__ == "__main__":
    main()
