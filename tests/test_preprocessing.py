"""
Test cases for data preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.preprocessing import (
    drop_inconsistent_features, drop_high_nan_features, filter_compliance_targets,
    process_categorical_features, one_hot_encode_features, engineer_features,
    fill_missing_values, preprocess_training_data, preprocess_test_data
)


class TestDataPreprocessing:
    """Test cases for data preprocessing functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'ticket_id': range(n_samples),
            'agency_name': ['Health'] * 500 + ['Building'] * 500,
            'inspector_name': [f'Inspector_{i%10}' for i in range(n_samples)],
            'violator_name': [f'Violator_{i%50}' for i in range(n_samples)],
            'violation_street_number': range(1000, 1000 + n_samples),
            'violation_street_name': [f'Street_{i%20}' for i in range(n_samples)],
            'mailing_address_str_number': range(2000, 2000 + n_samples),
            'mailing_address_str_name': [f'Mail_Street_{i%30}' for i in range(n_samples)],
            'city': ['Detroit'] * n_samples,
            'state': ['MI'] * n_samples,
            'zip_code': [f'4820{i%10}' for i in range(n_samples)],
            'country': ['USA'] * n_samples,
            'ticket_issued_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'hearing_date': pd.date_range('2020-02-01', periods=n_samples, freq='D'),
            'violation_code': [f'CODE_{i%5}' for i in range(n_samples)],
            'violation_description': [f'Description_{i%3}' for i in range(n_samples)],
            'disposition': [f'Disposition_{i%4}' for i in range(n_samples)],
            'fine_amount': np.random.uniform(50, 1000, n_samples),
            'admin_fee': [20.0] * n_samples,
            'state_fee': [10.0] * n_samples,
            'late_fee': np.random.uniform(0, 100, n_samples),
            'discount_amount': [0.0] * n_samples,
            'clean_up_cost': [0.0] * n_samples,
            'judgment_amount': np.random.uniform(80, 1200, n_samples),
            'payment_date': pd.date_range('2020-03-01', periods=n_samples, freq='D'),
            'payment_status': ['Paid'] * 500 + ['Unpaid'] * 500,
            'compliance': np.random.choice([0.0, 1.0, np.nan], n_samples, p=[0.6, 0.3, 0.1])
        }
        
        # Add some high-NaN columns
        data['high_nan_col'] = [np.nan] * int(0.8 * n_samples) + [1.0] * int(0.2 * n_samples)
        data['low_nan_col'] = [np.nan] * int(0.1 * n_samples) + [1.0] * int(0.9 * n_samples)
        
        df = pd.DataFrame(data)
        return df
    
    def test_drop_inconsistent_features(self, sample_data):
        """Test dropping inconsistent features."""
        original_cols = set(sample_data.columns)
        processed_df = drop_inconsistent_features(sample_data)
        remaining_cols = set(processed_df.columns)
        
        # Check that inconsistent columns were dropped
        inconsistent_cols = {'payment_date', 'payment_status', 'inspector_name', 'violator_name'}
        dropped_cols = original_cols - remaining_cols
        
        assert inconsistent_cols.issubset(dropped_cols)
        assert 'compliance' in remaining_cols  # Target should remain
        assert len(processed_df) == len(sample_data)  # Rows should remain same
    
    def test_drop_high_nan_features(self, sample_data):
        """Test dropping high-NaN features."""
        original_cols = set(sample_data.columns)
        processed_df = drop_high_nan_features(sample_data, threshold=0.5)
        remaining_cols = set(processed_df.columns)
        
        # Check that high-NaN column was dropped
        assert 'high_nan_col' not in remaining_cols
        assert 'low_nan_col' in remaining_cols  # Low-NaN should remain
        assert len(processed_df) == len(sample_data)  # Rows should remain same
    
    def test_filter_compliance_targets(self, sample_data):
        """Test filtering compliance targets."""
        original_count = len(sample_data)
        processed_df = filter_compliance_targets(sample_data)
        filtered_count = len(processed_df)
        
        # Should have fewer rows (NaN compliance removed)
        assert filtered_count < original_count
        assert processed_df['compliance'].notnull().all()  # No NaN compliance remaining
    
    def test_filter_compliance_no_target(self, sample_data):
        """Test filtering when no compliance column exists."""
        df_no_target = sample_data.drop('compliance', axis=1)
        processed_df = filter_compliance_targets(df_no_target)
        
        # Should return unchanged DataFrame
        assert len(processed_df) == len(df_no_target)
        assert 'compliance' not in processed_df.columns
    
    def test_process_categorical_features(self, sample_data):
        """Test processing categorical features."""
        # Remove inconsistent features first
        df = drop_inconsistent_features(sample_data)
        
        processed_df, categorical_cols = process_categorical_features(df)
        
        assert isinstance(categorical_cols, list)
        assert len(categorical_cols) > 0
        assert processed_df.shape[0] == df.shape[0]  # Same number of rows
        
        # Check that categorical columns are still present
        for col in categorical_cols:
            assert col in processed_df.columns
    
    def test_one_hot_encode_features(self, sample_data):
        """Test one-hot encoding."""
        # Process categorical features first
        df = drop_inconsistent_features(sample_data)
        df, categorical_cols = process_categorical_features(df)
        
        encoded_df = one_hot_encode_features(df, categorical_cols)
        
        # Should have more columns after one-hot encoding
        assert encoded_df.shape[1] > df.shape[1]
        assert len(encoded_df) == len(df)  # Same number of rows
        
        # Original categorical columns should be gone
        for col in categorical_cols:
            assert col not in encoded_df.columns
    
    def test_engineer_features(self, sample_data):
        """Test feature engineering."""
        processed_df = engineer_features(sample_data)
        
        # Check that date difference feature was created
        assert 'hearing_issued_date_diff' in processed_df.columns
        
        # Check that original date columns were dropped
        assert 'hearing_date' not in processed_df.columns
        assert 'ticket_issued_date' not in processed_df.columns
        
        # Check that the feature is numeric
        assert pd.api.types.is_numeric_dtype(processed_df['hearing_issued_date_diff'])
    
    def test_fill_missing_values(self, sample_data):
        """Test filling missing values."""
        # Add some NaN values to numeric columns
        df = sample_data.copy()
        df.loc[:10, 'fine_amount'] = np.nan
        df.loc[:5, 'admin_fee'] = np.nan
        
        filled_df = fill_missing_values(df)
        
        # Check that no NaN values remain
        assert not filled_df.isnull().any().any()
        
        # Check that the DataFrame structure is preserved
        assert filled_df.shape == df.shape
        assert list(filled_df.columns) == list(df.columns)
    
    def test_preprocess_training_data(self, sample_data):
        """Test complete training data preprocessing pipeline."""
        processed_df = preprocess_training_data(sample_data)
        
        # Should have compliance column
        assert 'compliance' in processed_df.columns
        
        # Should have no NaN values
        assert not processed_df.isnull().any().any()
        
        # Should have engineered features
        assert 'hearing_issued_date_diff' in processed_df.columns
        
        # Should not have inconsistent features
        assert 'payment_date' not in processed_df.columns
        assert 'inspector_name' not in processed_df.columns
        
        # All columns should be numeric (one-hot encoded)
        for col in processed_df.columns:
            if col != 'compliance':
                assert pd.api.types.is_numeric_dtype(processed_df[col])
    
    def test_preprocess_test_data(self, sample_data):
        """Test test data preprocessing."""
        # First preprocess training data
        train_df = preprocess_training_data(sample_data)
        training_columns = train_df.columns.tolist()
        
        # Create test data (without compliance)
        test_df = sample_data.drop('compliance', axis=1)
        
        # Preprocess test data
        processed_test_df = preprocess_test_data(test_df, training_columns)
        
        # Should not have compliance column
        assert 'compliance' not in processed_test_df.columns
        
        # Should have same number of features as training data (minus compliance)
        expected_features = len([col for col in training_columns if col != 'compliance'])
        assert processed_test_df.shape[1] == expected_features
        
        # Should have no NaN values
        assert not processed_test_df.isnull().any().any()


class TestDataIntegration:
    """Test cases for data integration and edge cases."""
    
    def test_empty_dataframe(self):
        """Test processing empty DataFrame."""
        df = pd.DataFrame()
        
        # Should not raise errors
        result = drop_inconsistent_features(df)
        assert len(result) == 0
    
    def test_single_row_dataframe(self):
        """Test processing single row DataFrame."""
        df = pd.DataFrame({
            'ticket_id': [1],
            'compliance': [1.0],
            'fine_amount': [100.0],
            'agency_name': ['Health']
        })
        
        processed_df = preprocess_training_data(df)
        assert len(processed_df) == 1
        assert 'compliance' in processed_df.columns
    
    def test_all_nan_column(self):
        """Test handling of completely NaN column."""
        df = pd.DataFrame({
            'ticket_id': [1, 2, 3],
            'compliance': [1.0, 0.0, 1.0],
            'all_nan_col': [np.nan, np.nan, np.nan],
            'fine_amount': [100.0, 200.0, 150.0]
        })
        
        # Should drop all-NaN column
        processed_df = drop_high_nan_features(df, threshold=0.5)
        assert 'all_nan_col' not in processed_df.columns
    
    def test_duplicate_columns(self):
        """Test handling of duplicate columns."""
        df = pd.DataFrame({
            'ticket_id': [1, 2, 3],
            'compliance': [1.0, 0.0, 1.0],
            'fine_amount': [100.0, 200.0, 150.0],
            'ticket_id': [1, 2, 3]  # Duplicate
        })
        
        # Should handle gracefully
        processed_df = preprocess_training_data(df)
        assert len(processed_df) == 3


if __name__ == "__main__":
    pytest.main([__file__])
