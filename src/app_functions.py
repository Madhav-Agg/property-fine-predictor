"""
Core application functions for Streamlit app.

This module contains reusable functions for data processing,
model training, and prediction that can be used in the Streamlit interface.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from src.preprocessing import load_data, merge_address_data, preprocess_training_data, preprocess_test_data
from src.models import BlightComplianceModel, compare_models
from src.utils import get_logger, MODEL_DIR, PROCESSED_DATA_DIR

logger = get_logger(__name__)


class PropertyFinesApp:
    """
    Main application class for Property Maintenance Fines Prediction.
    
    This class encapsulates all the core functionality needed for the Streamlit app,
    including data loading, model training, and prediction.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.models = {}
        self.training_data = None
        self.test_data = None
        self.address_latlons_df = None
        self.is_initialized = False
    
    def initialize_data(self) -> Dict[str, Any]:
        """
        Load and prepare all data for the application.
        
        Returns:
            Dictionary with initialization status and data info
        """
        try:
            logger.info("Initializing application data...")
            
            # Load raw data
            train_df, test_df, address_df, latlons_df = load_data()
            
            # Merge address and lat/lon data
            self.address_latlons_df = merge_address_data(address_df, latlons_df)
            
            # Merge location data with train and test data
            train_df = pd.merge(train_df, self.address_latlons_df, how='inner', on='ticket_id')
            test_df = pd.merge(test_df, self.address_latlons_df, how='inner', on='ticket_id')
            
            # Drop original location columns
            location_cols = ['violation_street_number', 'violation_street_name']
            train_df = train_df.drop(location_cols, axis=1)
            test_df = test_df.drop(location_cols, axis=1)
            
            # Preprocess data
            self.training_data = preprocess_training_data(train_df)
            
            # Get feature columns from training data (excluding target)
            training_columns = self.training_data.columns.tolist()
            self.test_data = preprocess_test_data(test_df, training_columns)
            
            # Load existing models
            self._load_existing_models()
            
            self.is_initialized = True
            
            return {
                'status': 'success',
                'message': 'Data initialized successfully',
                'train_samples': len(self.training_data),
                'test_samples': len(self.test_data),
                'features': len(self.training_data.columns) - 1,  # Exclude target
                'models_loaded': list(self.models.keys())
            }
            
        except Exception as e:
            logger.error(f"Error initializing data: {e}")
            return {
                'status': 'error',
                'message': f'Failed to initialize data: {str(e)}'
            }
    
    def _load_existing_models(self):
        """Load existing trained models."""
        for model_type in ['gradient_boosting', 'logistic_regression']:
            model_file = f"{model_type}_model.pkl"
            model_path = MODEL_DIR / model_file
            
            if model_path.exists():
                try:
                    model = BlightComplianceModel(model_type=model_type)
                    model.load_model(model_file)
                    self.models[model_type] = model
                    logger.info(f"Loaded {model_type} model")
                except Exception as e:
                    logger.error(f"Failed to load {model_type} model: {e}")
    
    def train_model(self, model_type: str, hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """
        Train a machine learning model.
        
        Args:
            model_type: Type of model to train
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary with training results
        """
        if not self.is_initialized:
            return {
                'status': 'error',
                'message': 'Application not initialized. Call initialize_data() first.'
            }
        
        try:
            logger.info(f"Training {model_type} model...")
            
            # Initialize model
            model = BlightComplianceModel(model_type=model_type)
            
            if hyperparameter_tuning:
                tuning_results = model.hyperparameter_tuning(self.training_data)
                metrics = {
                    'best_params': tuning_results['best_params'],
                    'best_score': tuning_results['best_score'],
                    'cv_mean': tuning_results['best_score'],
                    'cv_std': 0.0
                }
            else:
                training_metrics = model.train(self.training_data)
                metrics = {
                    'cv_mean': training_metrics['cv_mean'],
                    'cv_std': training_metrics['cv_std'],
                    'train_auc': training_metrics['train_auc'],
                    'train_accuracy': training_metrics['train_accuracy']
                }
            
            # Save model
            model_file = f"{model_type}_model.pkl"
            model.save_model(model_file)
            
            # Update models dictionary
            self.models[model_type] = model
            
            # Get feature importance if available
            importance = model.get_feature_importance()
            if importance is not None:
                metrics['top_features'] = importance.head(10).to_dict()
            
            return {
                'status': 'success',
                'message': f'{model_type} model trained successfully',
                'model_file': model_file,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {
                'status': 'error',
                'message': f'Failed to train model: {str(e)}'
            }
    
    def predict_single(self, model_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single ticket.
        
        Args:
            model_type: Type of model to use
            input_data: Dictionary with ticket features
        
        Returns:
            Dictionary with prediction result
        """
        if not self.is_initialized:
            return {
                'status': 'error',
                'message': 'Application not initialized. Call initialize_data() first.'
            }
        
        if model_type not in self.models:
            return {
                'status': 'error',
                'message': f'Model {model_type} not available. Train the model first.'
            }
        
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            
            # Preprocess the input data the same way as training data
            # Get the training columns (excluding target)
            training_columns = self.training_data.columns.tolist()
            training_columns = [col for col in training_columns if col != 'compliance']
            
            # Apply same preprocessing as test data
            processed_df = preprocess_test_data(df, training_columns)
            
            # Make prediction
            model = self.models[model_type]
            predictions = model.predict(processed_df)
            
            probability = float(predictions.iloc[0])
            prediction = 'compliant' if probability > 0.5 else 'non_compliant'
            
            return {
                'status': 'success',
                'probability': probability,
                'prediction': prediction,
                'confidence': 'high' if probability > 0.8 or probability < 0.2 else 'medium' if probability > 0.7 or probability < 0.3 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'status': 'error',
                'message': f'Failed to make prediction: {str(e)}'
            }
    
    def predict_batch(self, model_type: str, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Make predictions on test data.
        
        Args:
            model_type: Type of model to use
            sample_size: Optional sample size for faster testing
        
        Returns:
            Dictionary with batch prediction results
        """
        if not self.is_initialized:
            return {
                'status': 'error',
                'message': 'Application not initialized. Call initialize_data() first.'
            }
        
        if model_type not in self.models:
            return {
                'status': 'error',
                'message': f'Model {model_type} not available. Train the model first.'
            }
        
        try:
            # Sample data if requested
            test_data = self.test_data.copy()
            if sample_size and sample_size < len(test_data):
                test_data = test_data.head(sample_size)
            
            # Make predictions
            model = self.models[model_type]
            predictions = model.predict_with_ids(test_data)
            
            # Create summary statistics
            summary = {
                'total_predictions': len(predictions),
                'mean_probability': float(predictions.mean()),
                'std_probability': float(predictions.std()),
                'min_probability': float(predictions.min()),
                'max_probability': float(predictions.max()),
                'compliant_count': int((predictions > 0.5).sum()),
                'non_compliant_count': int((predictions <= 0.5).sum())
            }
            
            # Get sample predictions for display
            sample_predictions = predictions.head(10).to_dict()
            
            return {
                'status': 'success',
                'summary': summary,
                'sample_predictions': sample_predictions,
                'all_predictions': predictions.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            return {
                'status': 'error',
                'message': f'Failed to make batch predictions: {str(e)}'
            }
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get information about a trained model.
        
        Args:
            model_type: Type of model
        
        Returns:
            Dictionary with model information
        """
        if model_type not in self.models:
            return {
                'status': 'error',
                'message': f'Model {model_type} not available'
            }
        
        try:
            model = self.models[model_type]
            
            info = {
                'status': 'success',
                'model_type': model_type,
                'is_trained': model.is_trained,
                'feature_columns': len(model.feature_columns) if model.feature_columns else 0,
                'model_class': type(model.model).__name__
            }
            
            # Add feature importance if available
            importance = model.get_feature_importance()
            if importance is not None:
                info['top_features'] = importance.head(10).to_dict()
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                'status': 'error',
                'message': f'Failed to get model info: {str(e)}'
            }
    
    def compare_models(self) -> Dict[str, Any]:
        """
        Compare all available models.
        
        Returns:
            Dictionary with model comparison results
        """
        if not self.is_initialized:
            return {
                'status': 'error',
                'message': 'Application not initialized. Call initialize_data() first.'
            }
        
        try:
            results = compare_models(self.training_data)
            
            # Format results for display
            comparison = {}
            for model_type, metrics in results.items():
                comparison[model_type] = {
                    'cv_auc': f"{metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})",
                    'train_auc': f"{metrics['train_auc']:.4f}",
                    'train_accuracy': f"{metrics['train_accuracy']:.4f}",
                    'n_samples': metrics['n_samples'],
                    'n_features': metrics['n_features']
                }
                
                if 'feature_importance' in metrics:
                    comparison[model_type]['top_features'] = metrics['feature_importance'].head(5).to_dict()
            
            return {
                'status': 'success',
                'comparison': comparison
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {
                'status': 'error',
                'message': f'Failed to compare models: {str(e)}'
            }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded data.
        
        Returns:
            Dictionary with data summary
        """
        if not self.is_initialized:
            return {
                'status': 'error',
                'message': 'Application not initialized. Call initialize_data() first.'
            }
        
        try:
            # Training data summary
            train_summary = {
                'samples': len(self.training_data),
                'features': len(self.training_data.columns) - 1,  # Exclude target
                'compliance_rate': float(self.training_data['compliance'].mean()),
                'compliant_count': int(self.training_data['compliance'].sum()),
                'non_compliant_count': int(len(self.training_data) - self.training_data['compliance'].sum())
            }
            
            # Test data summary
            test_summary = {
                'samples': len(self.test_data),
                'features': len(self.test_data.columns)
            }
            
            return {
                'status': 'success',
                'training_data': train_summary,
                'test_data': test_summary,
                'available_models': list(self.models.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {
                'status': 'error',
                'message': f'Failed to get data summary: {str(e)}'
            }


# Global app instance
_app_instance = None


def get_app_instance() -> PropertyFinesApp:
    """Get or create the global app instance."""
    global _app_instance
    if _app_instance is None:
        _app_instance = PropertyFinesApp()
    return _app_instance


def create_sample_input() -> Dict[str, Any]:
    """
    Create a sample input for demonstration purposes.
    
    Returns:
        Dictionary with sample ticket data
    """
    return {
        'ticket_id': 12345,
        'agency_name': 'Department of Health',
        'violation_code': 'CODE_1',
        'violation_description': 'Failure to maintain exterior',
        'disposition': 'Responsible',
        'fine_amount': 250.0,
        'admin_fee': 20.0,
        'state_fee': 10.0,
        'late_fee': 25.0,
        'discount_amount': 0.0,
        'clean_up_cost': 0.0,
        'judgment_amount': 305.0,
        'lat': 42.3314,
        'lon': -83.0458,
        'hearing_issued_date_diff': 30
    }


def validate_input_data(input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate input data for prediction.
    
    Args:
        input_data: Dictionary with input features
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    required_fields = [
        'agency_name', 'violation_code', 'violation_description',
        'disposition', 'fine_amount', 'admin_fee', 'state_fee',
        'late_fee', 'discount_amount', 'clean_up_cost',
        'judgment_amount', 'lat', 'lon', 'hearing_issued_date_diff'
    ]
    
    for field in required_fields:
        if field not in input_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate numeric fields
    numeric_fields = [
        'fine_amount', 'admin_fee', 'state_fee', 'late_fee',
        'discount_amount', 'clean_up_cost', 'judgment_amount',
        'lat', 'lon', 'hearing_issued_date_diff'
    ]
    
    for field in numeric_fields:
        if field in input_data:
            try:
                value = float(input_data[field])
                if field in ['lat', 'lon'] and (value < -180 or value > 180):
                    errors.append(f"Invalid {field}: {value}")
                elif field in ['fine_amount', 'admin_fee', 'state_fee', 'late_fee',
                             'discount_amount', 'clean_up_cost', 'judgment_amount'] and value < 0:
                    errors.append(f"{field} cannot be negative: {value}")
                elif field == 'hearing_issued_date_diff' and value < 0:
                    errors.append(f"hearing_issued_date_diff cannot be negative: {value}")
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric value for {field}: {input_data[field]}")
    
    return len(errors) == 0, errors
