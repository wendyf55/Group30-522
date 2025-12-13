import click
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the function from src/download.py
from download import download_and_extract

@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")
@click.option('--write_to', type=str, help="Path to directory where raw data will be written to")
def main(url, write_to):
    """Downloads data zip from the web to a local filepath and extracts it."""
    try:
        # Try to download and extract directly
        download_and_extract(url, write_to)
    except FileNotFoundError:
        # If the directory doesn't exist, create it and try again
        os.makedirs(write_to, exist_ok=True)
        download_and_extract(url, write_to)

if __name__ == '__main__':
    main()
