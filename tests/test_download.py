import io
import zipfile
import pytest
import requests
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.download import download_and_extract


def _create_mock_zip() -> bytes:
    """Create an in-memory zip containing a single text file."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("test_file.txt", "This is a test file")
    return buffer.getvalue()


def test_download_and_extract_valid(tmp_path):
    """
    Test that download_and_extract saves the zip and extracts its contents.
    """
    mock_zip_content = _create_mock_zip()

    with patch("src.download.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = mock_zip_content
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        download_and_extract("http://fake-url.com/data.zip", str(tmp_path))

    expected_zip = tmp_path / "data.zip"
    expected_file = tmp_path / "test_file.txt"

    assert expected_zip.exists(), "Zip file was not written to disk."
    assert expected_file.exists(), "File from inside the zip was not extracted."


def test_download_and_extract_invalid_url(tmp_path):
    """
    Test that download_and_extract raises an error if the HTTP request fails.
    """
    with patch("src.download.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            download_and_extract("http://fake-url.com/bad_link.zip", str(tmp_path))