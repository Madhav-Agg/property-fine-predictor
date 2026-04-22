"""
Test cases for machine learning models.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import BlightComplianceModel, compare_models
from src.preprocessing import preprocess_training_data


class TestBlightComplianceModel:
    """Test cases for the BlightComplianceModel class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'ticket_id': range(n_samples),
            'agency_name': ['Health'] * 500 + ['Building'] * 500,
            'fine_amount': np.random.uniform(50, 1000, n_samples),
            'admin_fee': [20.0] * n_samples,
            'state_fee': [10.0] * n_samples,
            'late_fee': np.random.uniform(0, 100, n_samples),
            'discount_amount': [0.0] * n_samples,
            'clean_up_cost': [0.0] * n_samples,
            'judgment_amount': np.random.uniform(80, 1200, n_samples),
            'lat': np.random.uniform(42.0, 42.5, n_samples),
            'lon': np.random.uniform(-83.5, -83.0, n_samples),
            'hearing_issued_date_diff': np.random.randint(1, 365, n_samples),
            'compliance': np.random.choice([0.0, 1.0], n_samples, p=[0.7, 0.3])
        }
        
        df = pd.DataFrame(data)
        return df
    
    @pytest.fixture
    def model_gb(self):
        """Create a Gradient Boosting model instance."""
        return BlightComplianceModel(model_type='gradient_boosting')
    
    @pytest.fixture
    def model_lr(self):
        """Create a Logistic Regression model instance."""
        return BlightComplianceModel(model_type='logistic_regression')
    
    def test_model_initialization(self, model_gb, model_lr):
        """Test model initialization."""
        assert model_gb.model_type == 'gradient_boosting'
        assert model_lr.model_type == 'logistic_regression'
        assert not model_gb.is_trained
        assert not model_lr.is_trained
        assert model_gb.scaler is None
        assert model_lr.scaler is not None
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            BlightComplianceModel(model_type='invalid_model')
    
    def test_prepare_features_training(self, model_gb, sample_data):
        """Test feature preparation for training."""
        X, y = model_gb.prepare_features(sample_data, is_training=True)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == len(sample_data)
        assert 'compliance' not in X.columns
        assert model_gb.feature_columns is not None
        assert len(model_gb.feature_columns) == X.shape[1]
    
    def test_prepare_features_prediction(self, model_gb, sample_data):
        """Test feature preparation for prediction."""
        # First train to get feature columns
        X_train, y_train = model_gb.prepare_features(sample_data, is_training=True)
        model_gb.model.fit(X_train, y_train)
        model_gb.is_trained = True
        
        # Remove compliance column for prediction
        test_data = sample_data.drop('compliance', axis=1)
        X_test, y_test = model_gb.prepare_features(test_data, is_training=False)
        
        assert isinstance(X_test, pd.DataFrame)
        assert y_test is None
        assert len(X_test) == len(test_data)
    
    def test_train_model(self, model_gb, sample_data):
        """Test model training."""
        metrics = model_gb.train(sample_data)
        
        assert model_gb.is_trained
        assert 'cv_mean' in metrics
        assert 'cv_std' in metrics
        assert 'train_auc' in metrics
        assert 'train_accuracy' in metrics
        assert 0 <= metrics['cv_mean'] <= 1
        assert 0 <= metrics['train_auc'] <= 1
        assert 0 <= metrics['train_accuracy'] <= 1
    
    def test_predict_untrained(self, model_gb, sample_data):
        """Test that prediction fails on untrained model."""
        test_data = sample_data.drop('compliance', axis=1)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model_gb.predict(test_data)
    
    def test_predict_trained(self, model_gb, sample_data):
        """Test prediction on trained model."""
        # Train model
        model_gb.train(sample_data)
        
        # Make predictions
        test_data = sample_data.drop('compliance', axis=1)
        predictions = model_gb.predict(test_data)
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(test_data)
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_predict_with_ids(self, model_gb, sample_data):
        """Test prediction with ticket_id indexing."""
        # Train model
        model_gb.train(sample_data)
        
        # Make predictions
        test_data = sample_data.drop('compliance', axis=1)
        predictions = model_gb.predict_with_ids(test_data)
        
        assert isinstance(predictions, pd.Series)
        assert predictions.name == 'compliance'
        assert len(predictions) == len(test_data)
        assert all(predictions.index == test_data['ticket_id'])
    
    def test_feature_importance(self, model_gb, sample_data):
        """Test feature importance extraction."""
        # Train model
        model_gb.train(sample_data)
        
        # Get feature importance
        importance = model_gb.get_feature_importance()
        
        assert isinstance(importance, pd.Series)
        assert importance.name == 'importance'
        assert len(importance) == len(model_gb.feature_columns)
        assert all(importance >= 0)
    
    def test_feature_importance_logistic(self, model_lr, sample_data):
        """Test that feature importance is not available for logistic regression."""
        # Train model
        model_lr.train(sample_data)
        
        # Get feature importance
        importance = model_lr.get_feature_importance()
        
        assert importance is None
    
    def test_hyperparameter_tuning(self, model_gb, sample_data):
        """Test hyperparameter tuning."""
        # Use smaller parameter grid for testing
        param_grid = {
            'n_estimators': [10, 20],
            'learning_rate': [0.1]
        }
        
        results = model_gb.hyperparameter_tuning(sample_data, param_grid)
        
        assert model_gb.is_trained
        assert 'best_params' in results
        assert 'best_score' in results
        assert 'cv_results' in results
        assert isinstance(results['best_score'], float)
        assert 0 <= results['best_score'] <= 1


class TestModelComparison:
    """Test cases for model comparison functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data for testing."""
        np.random.seed(42)
        n_samples = 500  # Smaller for faster testing
        
        data = {
            'ticket_id': range(n_samples),
            'agency_name': ['Health'] * 250 + ['Building'] * 250,
            'fine_amount': np.random.uniform(50, 1000, n_samples),
            'admin_fee': [20.0] * n_samples,
            'state_fee': [10.0] * n_samples,
            'late_fee': np.random.uniform(0, 100, n_samples),
            'discount_amount': [0.0] * n_samples,
            'clean_up_cost': [0.0] * n_samples,
            'judgment_amount': np.random.uniform(80, 1200, n_samples),
            'lat': np.random.uniform(42.0, 42.5, n_samples),
            'lon': np.random.uniform(-83.5, -83.0, n_samples),
            'hearing_issued_date_diff': np.random.randint(1, 365, n_samples),
            'compliance': np.random.choice([0.0, 1.0], n_samples, p=[0.7, 0.3])
        }
        
        df = pd.DataFrame(data)
        return df
    
    def test_compare_models(self, sample_data):
        """Test model comparison functionality."""
        results = compare_models(sample_data)
        
        assert isinstance(results, dict)
        assert 'gradient_boosting' in results
        assert 'logistic_regression' in results
        
        for model_type, metrics in results.items():
            assert 'cv_mean' in metrics
            assert 'cv_std' in metrics
            assert 'train_auc' in metrics
            assert 'train_accuracy' in metrics
            assert 'n_samples' in metrics
            assert 'n_features' in metrics


if __name__ == "__main__":
    pytest.main([__file__])
