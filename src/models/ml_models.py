"""
Machine learning models for the Property Maintenance Fines Prediction project.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from src.utils import get_logger, MODEL_DIR, GRADIENT_BOOSTING_PARAMS, LOGISTIC_REGRESSION_PARAMS, CV_FOLDS, SCORING_METRIC

logger = get_logger(__name__)


class BlightComplianceModel:
    """
    A class to handle training, evaluation, and prediction for blight compliance models.
    """
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Initialize the model.
        
        Args:
            model_type: Type of model ('gradient_boosting' or 'logistic_regression')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**GRADIENT_BOOSTING_PARAMS)
            logger.info(f"Initialized Gradient Boosting model with params: {GRADIENT_BOOSTING_PARAMS}")
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
            self.scaler = StandardScaler()
            logger.info(f"Initialized Logistic Regression model with params: {LOGISTIC_REGRESSION_PARAMS}")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features and target for training or prediction.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data (includes target)
        
        Returns:
            Tuple of (features DataFrame, target Series or None)
        """
        logger.info(f"Preparing features for {'training' if is_training else 'prediction'}...")
        
        # Separate features and target for training data
        if is_training and 'compliance' in df.columns:
            target = df['compliance']
            features = df.drop(['compliance'], axis=1)
        else:
            target = None
            features = df.copy()
        
        # Store feature columns for later use
        if is_training:
            self.feature_columns = features.columns.tolist()
            logger.info(f"Stored {len(self.feature_columns)} feature columns")
        
        # Scale features for logistic regression
        if self.model_type == 'logistic_regression' and self.scaler is not None:
            if is_training:
                features_scaled = self.scaler.fit_transform(features)
                features = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
                logger.info("Fitted scaler and transformed features")
            else:
                features_scaled = self.scaler.transform(features)
                features = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
                logger.info("Transformed features using fitted scaler")
        
        return features, target
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            df: Training DataFrame with features and target
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare features and target
        X, y = self.prepare_features(df, is_training=True)
        
        if y is None:
            raise ValueError("Training data must include 'compliance' target column")
        
        # Perform cross-validation
        logger.info(f"Performing {CV_FOLDS}-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X, y, cv=CV_FOLDS, scoring=SCORING_METRIC)
        
        # Train the final model on all data
        logger.info("Training final model on all data...")
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = self.model.predict(X)
        
        train_auc = roc_auc_score(y, y_pred_proba)
        train_accuracy = accuracy_score(y, y_pred)
        
        metrics = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_auc': train_auc,
            'train_accuracy': train_accuracy,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        logger.info(f"Training completed. CV AUC: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        logger.info(f"Training AUC: {train_auc:.4f}, Accuracy: {train_accuracy:.4f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with features
        
        Returns:
            Series with predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions with {self.model_type} model...")
        
        # Prepare features
        X, _ = self.prepare_features(df, is_training=False)
        
        # Make predictions
        predictions = self.model.predict_proba(X)[:, 1]
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return pd.Series(predictions, index=df.index, name='compliance_probability')
    
    def predict_with_ids(self, df: pd.DataFrame) -> pd.Series:
        """
        Make predictions and return with ticket_id as index.
        
        Args:
            df: DataFrame with features and ticket_id
        
        Returns:
            Series with predictions indexed by ticket_id
        """
        predictions = self.predict(df)
        
        if 'ticket_id' in df.columns:
            predictions.index = df['ticket_id']
            predictions.name = 'compliance'
        
        return predictions
    
    def hyperparameter_tuning(self, df: pd.DataFrame, param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            df: Training DataFrame
            param_grid: Parameter grid for tuning. If None, uses default grids.
        
        Returns:
            Dictionary with tuning results
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_type}...")
        
        # Prepare features and target
        X, y = self.prepare_features(df, is_training=True)
        
        if y is None:
            raise ValueError("Training data must include 'compliance' target column")
        
        # Default parameter grids
        if param_grid is None:
            if self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            elif self.model_type == 'logistic_regression':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=CV_FOLDS, 
            scoring=SCORING_METRIC, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return results
    
    def save_model(self, filename: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filename: Name of the file to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_path = MODEL_DIR / filename
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filename: Name of the file to load the model from
        """
        model_path = MODEL_DIR / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance for tree-based models.
        
        Returns:
            Series with feature importance or None if not available
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_columns,
                name='importance'
            ).sort_values(ascending=False)
            
            logger.info("Feature importance extracted")
            return importance
        else:
            logger.warning("Feature importance not available for this model type")
            return None


def compare_models(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        df: Training DataFrame
    
    Returns:
        Dictionary with results for each model
    """
    logger.info("Comparing multiple models...")
    
    models = ['gradient_boosting', 'logistic_regression']
    results = {}
    
    for model_type in models:
        logger.info(f"Evaluating {model_type}...")
        
        model = BlightComplianceModel(model_type=model_type)
        metrics = model.train(df)
        results[model_type] = metrics
        
        # Get feature importance if available
        importance = model.get_feature_importance()
        if importance is not None:
            results[model_type]['feature_importance'] = importance.head(10)  # Top 10 features
    
    # Log comparison
    logger.info("Model comparison completed:")
    for model_type, metrics in results.items():
        logger.info(f"{model_type}: CV AUC = {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    
    return results
