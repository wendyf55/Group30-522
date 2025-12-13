import os
import requests
import zipfile

def download_and_extract(url, write_to):
    """
    Downloads a zip file from a URL and extracts it to a specified directory.
    
    Parameters
    ----------
    url : str
        The URL of the zip file to download.
    write_to : str
        The directory where the files should be extracted.
    """
    # Define the path for the zip file
    zip_path = os.path.join(write_to, "data.zip")

    # Download the file
    response = requests.get(url)
    response.raise_for_status() 

    # Write the zip file to the directory
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Extract the contents
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(write_to)
