import os
import sys
import pytest
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.cleaning import read_and_split


@pytest.fixture
def sample_abalone_csv(tmp_path):
    """Create a temporary valid Abalone CSV file"""
    data = [
        ["M", 0.5, 0.4, 0.1, 0.8, 0.3, 0.2, 0.1, 10],
        ["F", 0.6, 0.5, 0.2, 1.0, 0.4, 0.3, 0.2, 12],
        ["I", 0.4, 0.3, 0.1, 0.6, 0.2, 0.1, 0.1, 8],
        ["M", 0.55, 0.45, 0.15, 0.9, 0.35, 0.25, 0.15, 11],
        ["F", 0.65, 0.55, 0.2, 1.1, 0.45, 0.35, 0.25, 13],
    ]

    file_path = tmp_path / "abalone.csv"
    pd.DataFrame(data).to_csv(file_path, index=False, header=False)
    return file_path


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory."""
    return tmp_path / "output"


# Tests


def test_creates_output_files(sample_abalone_csv, output_dir):
    """Test that train and test csv files are created"""
    read_and_split(sample_abalone_csv, output_dir)

    train_path = output_dir / "abalone_train.csv"
    test_path = output_dir / "abalone_test.csv"

    assert train_path.exists()
    assert test_path.exists()


def test_train_test_split_ratio(sample_abalone_csv, output_dir):
    """Test that data is split approximately 80/20"""
    read_and_split(sample_abalone_csv, output_dir)

    train_df = pd.read_csv(output_dir / "abalone_train.csv")
    test_df = pd.read_csv(output_dir / "abalone_test.csv")

    total_rows = len(train_df) + len(test_df)

    assert total_rows == 5
    assert len(train_df) == 4
    assert len(test_df) == 1


def test_no_missing_values_in_output(sample_abalone_csv, output_dir):
    """Ensure no missing values appear in output files."""
    read_and_split(sample_abalone_csv, output_dir)

    train_df = pd.read_csv(output_dir / "abalone_train.csv")
    test_df = pd.read_csv(output_dir / "abalone_test.csv")

    assert not train_df.isnull().any().any()
    assert not test_df.isnull().any().any()


def test_output_column_count(sample_abalone_csv, output_dir):
    """Ensure output files preserve all columns."""
    read_and_split(sample_abalone_csv, output_dir)

    train_df = pd.read_csv(output_dir / "abalone_train.csv")
    test_df = pd.read_csv(output_dir / "abalone_test.csv")

    assert train_df.shape[1] == 9
    assert test_df.shape[1] == 9


def test_missing_input_file_raises_error(output_dir):
    """Test that a missing input file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_and_split("non_existent_file.csv", output_dir)
