import os
import requests
import zipfile
import click


DEFAULT_URL = "https://archive.ics.uci.edu/static/public/1/abalone.zip"
DEFAULT_WRITE_TO = "../data/raw"


def download_and_extract(url, write_to): 
    """
    Download the Abalone zip file from `url` and extract it into `write_to`, 
    with a configurable output directory.
    """

    os.makedirs(write_to, exist_ok=True)

    zip_path = os.path.join(write_to, "abalone.zip")

    request = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(request.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(write_to)
   
@click.command()
@click.option(
    "--url",
    type=str,
    default=DEFAULT_URL,
    show_default=True,
    help="URL of dataset to be downloaded",
)
@click.option(
    "--write_to",
    type=str,
    default=DEFAULT_WRITE_TO,
    show_default=True,
    help="Path to directory where raw data will be written to",
)
def main(url, write_to):
    """Downloads Abalone data zip from the web and extracts it."""
    download_and_extract(url, write_to)

  
if __name__ == "__main__":
    main()

