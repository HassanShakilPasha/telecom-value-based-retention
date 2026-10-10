"""
Configuration and constants for the Telecom Customer Churn AI Project.
Course: AI2002 Artificial Intelligence (Track A - R&D, Group F26-11)
"""

from pathlib import Path
import logging

# Random Seed for Reproducibility
SEED = 42

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "Telecom+Customer+Churn"
RAW_DATA_PATH = DATA_DIR / "telecom_customer_churn.csv"
DATA_DICT_PATH = DATA_DIR / "telecom_data_dictionary.csv"
ZIPCODE_POPULATION_PATH = DATA_DIR / "telecom_zipcode_population.csv"

# Processed Data Outputs
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

# Target Variable Specification
TARGET_RAW_COL = "Customer Status"
TARGET_BINARY_COL = "Churn"
ID_COL = "Customer ID"

# Target Leakage Columns (STRICTLY EXCLUDED from training features)
LEAKAGE_COLS = [
    "Churn Category",
    "Churn Reason",
]

# Service-Dependent Features mapping (Null if parent service is No)
INTERNET_DEPENDENT_COLS = [
    "Internet Type",
    "Avg Monthly GB Download",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
]

PHONE_DEPENDENT_COLS = [
    "Multiple Lines",
    "Avg Monthly Long Distance Charges",
]

# Demographic, Protected, and Grouping Columns
DEMOGRAPHIC_COLS = [
    "Gender",
    "Age",
    "Married",
    "Number of Dependents",
]

LOCATION_COLS = [
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
]

FINANCIAL_COLS = [
    "Monthly Charge",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Revenue",
]

CONTRACT_COLS = [
    "Tenure in Months",
    "Offer",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

# Setup Logging
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("TelecomChurn")
