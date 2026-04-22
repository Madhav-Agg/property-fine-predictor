"""
Configuration file for the Property Maintenance Fines Prediction project.
"""

import os
import sys
from pathlib import Path

# Project root directory - handle different execution contexts
if hasattr(sys, '_MEIPASS'):
    # PyInstaller context
    PROJECT_ROOT = Path(sys._MEIPASS)
else:
    # Normal execution - find project root from this file
    PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data file names
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
ADDRESSES_FILE = "addresses.csv"
LATLONS_FILE = "latlons.csv"

# Model parameters
CATEGORICAL_THRESHOLD = 100
UNKNOWN_CATEGORY = "<unknown>"

# Model hyperparameters
GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'random_state': 42
}

LOGISTIC_REGRESSION_PARAMS = {
    'random_state': 42,
    'max_iter': 1000
}

# Cross-validation
CV_FOLDS = 5
SCORING_METRIC = 'roc_auc'

# Features to drop
INCONSISTENT_LABELS = [
    'payment_date', 'payment_status', 'collection_status', 
    'compliance_detail', 'balance_due', 'payment_amount'
]

DATA_LEAKAGE_LABELS = ['violator_name', 'inspector_name']

MAILING_ADDRESS_LABELS = [
    'mailing_address_str_number', 'mailing_address_str_name', 
    'city', 'state', 'zip_code', 'country'
]

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
