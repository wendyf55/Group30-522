import click
import os
import pandas as pd

DEFAULT_TRAIN_PATH = '../data/processed/abalone_train.csv'
DEFAULT_SAVE_DIR = '../results/eda'
DEFAULT_TABLE_FILE = 'descriptive_stats.csv'


def save_descriptive_table(train_df, descriptive_path):
    train_df.describe().round(2).to_csv(descriptive_path)


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
@click.option('--table-name',
              type=str,
              default=DEFAULT_TABLE_FILE,
              help='Filename to save descriptive stats including extension')
def main(train_path, save_dir, table_name):
    os.makedirs(save_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)

    save_descriptive_table(train_df, os.path.join(save_dir, table_name))


if __name__ == "__main__":
    main()
