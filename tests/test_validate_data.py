import pytest
import numpy as np
import pandas as pd
import pandera.pandas as pa
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.validate_data import validate_data


# Test normal case
valid_data = pd.DataFrame({
    "Sex": ['M', 'F', 'F'],
    "Length": [1.2, 3.4, 5.6],
    "Diameter": [1.2, 3.4, 5.6],
    "Height": [1.2, 3.4, 5.6],
    "Whole_weight": [1.2, 3.4, 5.6],
    "Shucked_weight": [1.2, 3.4, 5.6],
    "Viscera_weight": [1.2, 3.4, 5.6],
    "Shell_weight": [1.2, 3.4, 5.6],
    "Rings": [5, 2, 2]
})


# Test that non-dataframe input type raises TypeError
np_data = valid_data.copy().to_numpy()
def test_input_type():
    with pytest.raises(TypeError):
        validate_data(np_data)


# Test that empty input raises ValueError
empty_data = valid_data.copy().iloc[0:0]
def test_empty_dataframe():
    with pytest.raises(ValueError):
        validate_data(empty_data)


# Test that missing target raises SchemaError
missing_column = valid_data.copy().drop(columns='Rings')
def test_missing_columns():
    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(missing_column)


# Test that wrong data type in target raises SchemaError
wrong_dtype = valid_data.copy()
wrong_dtype.Rings = wrong_dtype.Rings.astype('object')
def test_wrong_dtype():
    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(wrong_dtype)


# Test that nulls in target raises SchemaError
nulls_data = valid_data.copy()
nulls_data.loc[0, 'Rings'] = np.nan
def test_null_values():
    # errors if columns contain null
    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(nulls_data)
