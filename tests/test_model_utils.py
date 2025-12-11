"""
Unit tests for src/model_utils.py
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model_utils import create_preprocessor


class TestCreatePreprocessor:
    """Tests for the create_preprocessor function."""

    def test_returns_column_transformer(self):
        """Test that create_preprocessor returns a ColumnTransformer object."""
        numeric_features = ['Length', 'Diameter']
        categorical_features = ['Sex']
        
        preprocessor = create_preprocessor(numeric_features, categorical_features)
        
        assert isinstance(preprocessor, ColumnTransformer)

    def test_preprocessor_has_correct_transformers(self):
        """Test that preprocessor has StandardScaler and OneHotEncoder."""
        numeric_features = ['Length', 'Diameter']
        categorical_features = ['Sex']
        
        preprocessor = create_preprocessor(numeric_features, categorical_features)
        
        # Check transformer names
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert 'standardscaler' in transformer_names
        assert 'onehotencoder' in transformer_names

    def test_preprocessor_transforms_data_correctly(self):
        """Test that preprocessor correctly transforms sample data."""
        # Create sample data
        df = pd.DataFrame({
            'Length': [0.5, 0.6, 0.7],
            'Diameter': [0.4, 0.5, 0.6],
            'Sex': ['M', 'F', 'I']
        })
        
        numeric_features = ['Length', 'Diameter']
        categorical_features = ['Sex']
        
        preprocessor = create_preprocessor(numeric_features, categorical_features)
        transformed = preprocessor.fit_transform(df)
        
        # Check output shape: 2 numeric + 2 categorical (one dropped due to 'if_binary' being False for 3 categories)
        # With 3 categories (M, F, I), OneHotEncoder with drop='if_binary' keeps all 3
        assert transformed.shape[0] == 3  # 3 rows
        assert transformed.shape[1] == 5  # 2 numeric + 3 one-hot encoded

    def test_numeric_features_are_scaled(self):
        """Test that numeric features are standardized (mean=0, std=1)."""
        df = pd.DataFrame({
            'Length': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Diameter': [10.0, 20.0, 30.0, 40.0, 50.0],
            'Sex': ['M', 'M', 'F', 'F', 'I']
        })
        
        numeric_features = ['Length', 'Diameter']
        categorical_features = ['Sex']
        
        preprocessor = create_preprocessor(numeric_features, categorical_features)
        transformed = preprocessor.fit_transform(df)
        
        # First two columns are the scaled numeric features
        scaled_numeric = transformed[:, :2]
        
        # Check that scaled features have mean close to 0 and std close to 1
        assert np.abs(scaled_numeric[:, 0].mean()) < 0.01
        assert np.abs(scaled_numeric[:, 1].mean()) < 0.01
        assert np.abs(scaled_numeric[:, 0].std() - 1.0) < 0.01
        assert np.abs(scaled_numeric[:, 1].std() - 1.0) < 0.01

    def test_categorical_features_are_one_hot_encoded(self):
        """Test that categorical features are one-hot encoded."""
        df = pd.DataFrame({
            'Length': [0.5, 0.6, 0.7],
            'Sex': ['M', 'F', 'I']
        })
        
        numeric_features = ['Length']
        categorical_features = ['Sex']
        
        preprocessor = create_preprocessor(numeric_features, categorical_features)
        transformed = preprocessor.fit_transform(df)
        
        # One-hot encoded columns should be 0 or 1
        one_hot_cols = transformed[:, 1:]  # Skip the first numeric column
        assert np.all((one_hot_cols == 0) | (one_hot_cols == 1))
        
        # Each row should have exactly one 1 in the one-hot columns
        assert np.all(one_hot_cols.sum(axis=1) == 1)

    def test_empty_numeric_features(self):
        """Test preprocessor with empty numeric features list."""
        df = pd.DataFrame({
            'Sex': ['M', 'F', 'I']
        })
        
        numeric_features = []
        categorical_features = ['Sex']
        
        preprocessor = create_preprocessor(numeric_features, categorical_features)
        transformed = preprocessor.fit_transform(df)
        
        # Should only have one-hot encoded columns
        assert transformed.shape[1] == 3  # 3 categories

    def test_empty_categorical_features(self):
        """Test preprocessor with empty categorical features list."""
        df = pd.DataFrame({
            'Length': [0.5, 0.6, 0.7],
            'Diameter': [0.4, 0.5, 0.6]
        })
        
        numeric_features = ['Length', 'Diameter']
        categorical_features = []
        
        preprocessor = create_preprocessor(numeric_features, categorical_features)
        transformed = preprocessor.fit_transform(df)
        
        # Should only have scaled numeric columns
        assert transformed.shape[1] == 2
