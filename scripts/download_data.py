# download_data.py
# authors: Group30
# date: 2025-12-02

import os
import zipfile
import requests
import click
import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_and_extract_zip(url, out_dir, zip_name="abalone.zip"):
    """
    Download a ZIP file from the given URL and extract it into out_dir.

    Parameters
    ----------
    url : str
        URL to the abalone.zip file.
    out_dir : str
        Directory where the ZIP and extracted files will be stored.
    zip_name : str, optional
        Name of the ZIP file to save as, by default "abalone.zip".
    """

    os.makedirs(out_dir, exist_ok=True)

    zip_path = os.path.join(out_dir, zip_name)

    request = requests.get(url)
    if request.status_code != 200:
        raise ValueError(
            f"Failed to download file from {url}. "
            f"Status code: {request.status_code}"
        )

    with open(zip_path, "wb") as f:
        f.write(request.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)

    print(f"Downloaded and extracted ZIP to: {out_dir}")


# ---------------------------------------------------------------------
# Helper function to fetch dataset via ucimlrepo and save as CSV
# ---------------------------------------------------------------------
def fetch_and_save_abalone_csv(out_dir, csv_name="abalone.csv"):
    """
    Fetch the Abalone dataset using ucimlrepo and save it as a CSV.

    Parameters
    ----------
    out_dir : str
        Directory where the CSV will be stored.
    csv_name : str, optional
        Name of the CSV file, by default "abalone.csv".
    """

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, csv_name)

    # fetch dataset (same as in your notebook)
    abalone = fetch_ucirepo(id=1)

    # Extract features and targets and combine into a single DataFrame
    X = abalone.data.features
    y = abalone.data.targets

    df = X.copy()
    # Assuming the target column is named "Rings"
    df["Rings"] = y

    df.to_csv(csv_path, index=False)
    print(f"Saved cleaned Abalone CSV to: {csv_path}")

@click.command()
@click.option(
    "--url",
    type=str,
    default="https://archive.ics.uci.edu/static/public/1/abalone.zip",
    show_default=True,
    help="URL to the abalone.zip file from the UCI repository.",
)
@click.option(
    "--out_dir",
    type=str,
    default="data/raw",
    show_default=True,
    help="Directory where the data will be stored.",
)
def main(url, out_dir):
    """
    Download the Abalone dataset ZIP, extract it, and also save a CSV
    version of the dataset using ucimlrepo.
    """

    download_and_extract_zip(url=url, out_dir=out_dir)

    fetch_and_save_abalone_csv(out_dir=out_dir)


if __name__ == "__main__":
    main()

