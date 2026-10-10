"""
Data Ingestion and Loading Module.
Loads the Maven Analytics California Telecom Churn dataset and merges Zipcode Population metadata.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import logging

from src.utils.config import (
    RAW_DATA_PATH,
    ZIPCODE_POPULATION_PATH,
    TARGET_RAW_COL,
    setup_logging,
)

logger = setup_logging()

def load_raw_dataset(
    data_path: Path = RAW_DATA_PATH,
    zipcode_path: Path = ZIPCODE_POPULATION_PATH,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw telecom churn data and merges with zipcode population table.
    
    Returns:
        df_merged (pd.DataFrame): Joined customer records.
        summary (dict): Ingestion summary statistics and checks.
    """
    logger.info(f"Loading raw telecom churn dataset from {data_path}...")
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")
        
    df_customers = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df_customers)} customer rows with {df_customers.shape[1]} columns.")
    
    # Ingest Zipcode population lookup
    if zipcode_path.exists():
        logger.info(f"Loading zipcode population data from {zipcode_path}...")
        df_zip = pd.read_csv(zipcode_path)
        # Ensure Zip Code column is typed consistently
        df_customers["Zip Code"] = df_customers["Zip Code"].astype(str).str.strip()
        df_zip["Zip Code"] = df_zip["Zip Code"].astype(str).str.strip()
        
        # Merge population
        df_merged = df_customers.merge(df_zip, on="Zip Code", how="left")
        logger.info(f"Merged population lookup. New shape: {df_merged.shape}")
    else:
        logger.warning(f"Zipcode population file not found at {zipcode_path}. Continuing without merge.")
        df_merged = df_customers.copy()

    # Integrity validation
    target_counts = df_merged[TARGET_RAW_COL].value_counts().to_dict()
    null_counts = df_merged.isna().sum().to_dict()
    
    summary = {
        "total_records": len(df_merged),
        "total_columns": df_merged.shape[1],
        "target_distribution": target_counts,
        "columns": list(df_merged.columns),
        "missing_values_count": {k: v for k, v in null_counts.items() if v > 0},
    }
    
    logger.info(f"Data ingestion complete. Target distribution: {target_counts}")
    return df_merged, summary

if __name__ == "__main__":
    df, summary = load_raw_dataset()
    print("Dataset Summary:")
    print(f"Total Rows: {summary['total_records']}")
    print(f"Total Columns: {summary['total_columns']}")
    print(f"Target Distribution: {summary['target_distribution']}")
