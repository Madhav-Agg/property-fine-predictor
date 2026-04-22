"""
Data loading utilities for the Property Maintenance Fines Prediction project.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
from src.utils import get_logger, RAW_DATA_DIR, TRAIN_FILE, TEST_FILE, ADDRESSES_FILE, LATLONS_FILE

logger = get_logger(__name__)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all required data files for the project.
    
    Returns:
        Tuple containing (train_df, test_df, address_df, latlons_df)
    
    Raises:
        FileNotFoundError: If any required data file is missing
    """
    logger.info("Loading data files...")
    
    # Define file paths
    train_path = RAW_DATA_DIR / TRAIN_FILE
    test_path = RAW_DATA_DIR / TEST_FILE
    address_path = RAW_DATA_DIR / ADDRESSES_FILE
    latlons_path = RAW_DATA_DIR / LATLONS_FILE
    
    # Check if files exist and provide helpful error messages
    missing_files = []
    for path, name in [(train_path, "train"), (test_path, "test"), 
                       (address_path, "addresses"), (latlons_path, "latlons")]:
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        error_msg = "Missing data files:\n" + "\n".join(missing_files)
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load data with proper encoding and date parsing
    dtypes = {'ticket_issued_date': 'str', 'hearing_date': 'str'}
    parse_dates = ['ticket_issued_date', 'hearing_date']
    
    try:
        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path, encoding="ISO-8859-1", 
                               dtype=dtypes, parse_dates=parse_dates, 
                               low_memory=False)
        logger.info(f"Loaded {len(train_df)} training samples")
        
        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path, encoding="ISO-8859-1", 
                              dtype=dtypes, parse_dates=parse_dates, 
                              low_memory=False)
        logger.info(f"Loaded {len(test_df)} test samples")
        
        logger.info(f"Loading address data from {address_path}")
        address_df = pd.read_csv(address_path, encoding="ISO-8859-1", 
                                low_memory=False)
        logger.info(f"Loaded {len(address_df)} address records")
        
        logger.info(f"Loading lat/lon data from {latlons_path}")
        latlons_df = pd.read_csv(latlons_path, encoding="ISO-8859-1", 
                                low_memory=False)
        logger.info(f"Loaded {len(latlons_df)} lat/lon records")
        
        logger.info("All data files loaded successfully")
        return train_df, test_df, address_df, latlons_df
        
    except Exception as e:
        logger.error(f"Error loading data files: {e}")
        raise


def merge_address_data(address_df: pd.DataFrame, latlons_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge address and lat/lon data on address column.
    
    Args:
        address_df: DataFrame containing ticket_id and address
        latlons_df: DataFrame containing address and lat/lon coordinates
    
    Returns:
        Merged DataFrame with ticket_id, lat, and lon columns
    """
    logger.info("Merging address and lat/lon data...")
    
    try:
        # Merge on address
        merged_df = pd.merge(address_df, latlons_df, how='inner', on='address')
        
        # Drop address column as it's no longer needed
        merged_df.drop('address', axis=1, inplace=True)
        
        logger.info(f"Address data merged successfully. Shape: {merged_df.shape}")
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging address data: {e}")
        raise
