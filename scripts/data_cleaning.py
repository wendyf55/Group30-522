import click
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from cleaning import read_and_split

DEFAULT_ORIGIN_PATH = "data/raw/abalone.data"
DEFAULT_OUTPUT_DIR = "data/processed"


@click.command()
@click.option(
    "--origin_path",
    default=DEFAULT_ORIGIN_PATH,
    show_default=True,
    help="Path to raw Abalone data file",
)
@click.option(
    "--output_dir",
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory where processed data will be written",
)
def main(origin_path, output_dir):
    """Reads, cleans, and splits Abalone data."""
    read_and_split(origin_path, output_dir)


if __name__ == "__main__":
    main()
