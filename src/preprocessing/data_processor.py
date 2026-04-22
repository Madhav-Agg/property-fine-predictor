"""
Data processing utilities for the Property Maintenance Fines Prediction project.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from src.utils import (
    get_logger, INCONSISTENT_LABELS, DATA_LEAKAGE_LABELS, 
    MAILING_ADDRESS_LABELS, CATEGORICAL_THRESHOLD, UNKNOWN_CATEGORY
)

logger = get_logger(__name__)


def drop_inconsistent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop features that are inconsistent between train and test sets or cause data leakage.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with inconsistent features dropped
    """
    logger.info("Dropping inconsistent and data leakage features...")
    
    labels_to_remove = INCONSISTENT_LABELS + DATA_LEAKAGE_LABELS
    
    # Only drop columns that exist in the DataFrame
    existing_labels = [col for col in labels_to_remove if col in df.columns]
    
    if existing_labels:
        df = df.drop(existing_labels, axis=1)
        logger.info(f"Dropped columns: {existing_labels}")
    else:
        logger.info("No inconsistent columns found to drop")
    
    return df


def drop_high_nan_features(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop features with high percentage of NaN values.
    
    Args:
        df: Input DataFrame
        threshold: Maximum allowed NaN ratio (default: 0.5)
    
    Returns:
        DataFrame with high-NaN features dropped
    """
    logger.info(f"Dropping features with >{threshold*100}% NaN values...")
    
    # Calculate NaN ratios
    nan_ratios = df.isnull().sum() / len(df)
    high_nan_cols = nan_ratios[nan_ratios > threshold].index.tolist()
    
    if high_nan_cols:
        df = df.drop(high_nan_cols, axis=1)
        logger.info(f"Dropped high-NaN columns: {high_nan_cols}")
    else:
        logger.info("No high-NaN columns found to drop")
    
    return df


def filter_compliance_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out rows where compliance is NaN (not responsible violations).
    
    Args:
        df: Input DataFrame with compliance column
    
    Returns:
        DataFrame with only valid compliance targets
    """
    logger.info("Filtering rows with valid compliance targets...")
    
    if 'compliance' not in df.columns:
        logger.warning("Compliance column not found in DataFrame")
        return df
    
    initial_count = len(df)
    df = df[df['compliance'].notnull()].copy()
    final_count = len(df)
    
    logger.info(f"Filtered {initial_count - final_count} rows with NaN compliance")
    logger.info(f"Remaining rows: {final_count}")
    
    return df


def process_categorical_features(df: pd.DataFrame, 
                               threshold: int = CATEGORICAL_THRESHOLD) -> Tuple[pd.DataFrame, List[str]]:
    """
    Process categorical features by grouping rare categories into 'unknown'.
    
    Args:
        df: Input DataFrame
        threshold: Minimum frequency for a category to be kept separate
    
    Returns:
        Tuple of (processed DataFrame, list of categorical column names)
    """
    logger.info("Processing categorical features...")
    
    # Get object columns
    categorical_cols = df.dtypes[df.dtypes == 'object'].index.tolist()
    logger.info(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")
    
    for col in categorical_cols:
        if col in df.columns:
            # Count category frequencies
            value_counts = df[col].value_counts()
            
            # Identify categories to keep
            categories_to_keep = value_counts[value_counts > threshold].index.tolist()
            
            # Convert to categorical and group rare categories
            df[col] = pd.Categorical(df[col], categories=categories_to_keep, ordered=True)
            df[col] = df[col].cat.add_categories(UNKNOWN_CATEGORY).fillna(UNKNOWN_CATEGORY)
            
            logger.info(f"Processed {col}: kept {len(categories_to_keep)} categories, grouped {len(value_counts) - len(categories_to_keep)} as {UNKNOWN_CATEGORY}")
    
    return df, categorical_cols


def one_hot_encode_features(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical features.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
    
    Returns:
        DataFrame with one-hot encoded features
    """
    logger.info("Applying one-hot encoding...")
    
    # Only encode columns that still exist
    existing_cols = [col for col in categorical_cols if col in df.columns]
    
    if existing_cols:
        initial_shape = df.shape
        df = pd.get_dummies(df, columns=existing_cols)
        final_shape = df.shape
        
        logger.info(f"One-hot encoded {len(existing_cols)} columns")
        logger.info(f"Shape changed from {initial_shape} to {final_shape}")
    else:
        logger.info("No categorical columns found for one-hot encoding")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from existing ones.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering features...")
    
    # Engineer date difference feature
    if 'hearing_date' in df.columns and 'ticket_issued_date' in df.columns:
        logger.info("Creating hearing_issued_date_diff feature...")
        df['hearing_issued_date_diff'] = (df['hearing_date'] - df['ticket_issued_date']).dt.days
        
        # Drop original date columns
        df = df.drop(['hearing_date', 'ticket_issued_date'], axis=1)
        logger.info("Dropped original date columns")
    else:
        logger.warning("Date columns not found for feature engineering")
    
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with missing values filled
    """
    logger.info("Filling missing values...")
    
    # Find columns with NaN values
    nan_cols = df.columns[df.isnull().any()].tolist()
    
    if nan_cols:
        logger.info(f"Found {len(nan_cols)} columns with NaN values: {nan_cols}")
        
        # Fill NaN values with mean for numeric columns
        for col in nan_cols:
            if df[col].dtype in ['float64', 'int64']:
                mean_value = df[col].mean()
                df[col] = df[col].fillna(mean_value)
                logger.info(f"Filled {col} with mean value: {mean_value:.4f}")
            else:
                # For categorical columns, fill with mode
                mode_value = df[col].mode()[0] if not df[col].mode().empty else UNKNOWN_CATEGORY
                df[col] = df[col].fillna(mode_value)
                logger.info(f"Filled {col} with mode value: {mode_value}")
    else:
        logger.info("No missing values found")
    
    return df


def preprocess_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for training data.
    
    Args:
        df: Raw training DataFrame
    
    Returns:
        Fully processed training DataFrame
    """
    logger.info("Starting training data preprocessing pipeline...")
    
    # Step 1: Drop inconsistent and data leakage features
    df = drop_inconsistent_features(df)
    
    # Step 2: Drop high-NaN features
    df = drop_high_nan_features(df)
    
    # Step 3: Filter compliance targets
    df = filter_compliance_targets(df)
    
    # Step 4: Drop mailing address columns
    existing_mail_cols = [col for col in MAILING_ADDRESS_LABELS if col in df.columns]
    if existing_mail_cols:
        df = df.drop(existing_mail_cols, axis=1)
        logger.info(f"Dropped mailing address columns: {existing_mail_cols}")
    
    # Step 5: Process categorical features
    df, categorical_cols = process_categorical_features(df)
    
    # Step 6: One-hot encode
    df = one_hot_encode_features(df, categorical_cols)
    
    # Step 7: Engineer features
    df = engineer_features(df)
    
    # Step 8: Fill missing values
    df = fill_missing_values(df)
    
    logger.info("Training data preprocessing completed")
    logger.info(f"Final shape: {df.shape}")
    
    return df


def preprocess_test_data(df: pd.DataFrame, training_columns: List[str]) -> pd.DataFrame:
    """
    Preprocess test data to match training data structure.
    
    Args:
        df: Raw test DataFrame
        training_columns: List of columns from training data
    
    Returns:
        Processed test DataFrame matching training structure
    """
    logger.info("Starting test data preprocessing pipeline...")
    
    # Apply same preprocessing steps as training data (except compliance filtering)
    df = drop_inconsistent_features(df)
    df = drop_high_nan_features(df)
    
    # Drop mailing address columns
    existing_mail_cols = [col for col in MAILING_ADDRESS_LABELS if col in df.columns]
    if existing_mail_cols:
        df = df.drop(existing_mail_cols, axis=1)
    
    # Process categorical features
    df, categorical_cols = process_categorical_features(df)
    
    # One-hot encode
    df = one_hot_encode_features(df, categorical_cols)
    
    # Engineer features
    df = engineer_features(df)
    
    # Fill missing values
    df = fill_missing_values(df)
    
    # Ensure test data has same columns as training data
    missing_cols = set(training_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(training_columns)
    
    # Add missing columns with zeros
    for col in missing_cols:
        if col != 'compliance':  # Don't add compliance column to test data
            df[col] = 0
            logger.info(f"Added missing column to test data: {col}")
    
    # Remove extra columns (except ticket_id)
    extra_cols_to_remove = [col for col in extra_cols if col != 'ticket_id']
    if extra_cols_to_remove:
        df = df.drop(extra_cols_to_remove, axis=1)
        logger.info(f"Removed extra columns from test data: {extra_cols_to_remove}")
    
    # Reorder columns to match training data (excluding compliance)
    test_columns = [col for col in training_columns if col != 'compliance' and col in df.columns]
    df = df[test_columns]
    
    logger.info("Test data preprocessing completed")
    logger.info(f"Final shape: {df.shape}")
    
    return df
