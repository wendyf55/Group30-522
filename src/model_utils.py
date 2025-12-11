"""
Utility functions for model preprocessing and evaluation.
"""

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(numeric_features, categorical_features):
    """
    Create a preprocessing pipeline for numeric and categorical features.

    This function builds a scikit-learn ColumnTransformer that applies
    StandardScaler to numeric features and OneHotEncoder to categorical features.

    Parameters
    ----------
    numeric_features : list of str
        List of column names for numeric features to be scaled.
    categorical_features : list of str
        List of column names for categorical features to be one-hot encoded.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        A fitted ColumnTransformer that can be used in a scikit-learn pipeline.

    Examples
    --------
    >>> numeric_features = ['Length', 'Diameter', 'Height']
    >>> categorical_features = ['Sex']
    >>> preprocessor = create_preprocessor(numeric_features, categorical_features)
    >>> preprocessor
    ColumnTransformer(...)
    """
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(drop='if_binary', sparse_output=False), categorical_features)
    )
    return preprocessor
