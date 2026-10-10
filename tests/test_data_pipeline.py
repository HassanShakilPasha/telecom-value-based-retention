"""
Unit and Integration Tests for Phase 2 Data Ingestion, Cleaning & Preprocessing Pipeline.
Course: AI2002 Artificial Intelligence (Track A - R&D, Group F26-11)
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.config import (
    RAW_DATA_PATH,
    ZIPCODE_POPULATION_PATH,
    TARGET_BINARY_COL,
    LEAKAGE_COLS,
    ID_COL,
)
from src.data.loader import load_raw_dataset
from src.data.preprocessor import (
    clean_and_engineer_features,
    split_and_preprocess_dataset,
)

class TestPhase2DataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads and precomputes artifacts once for all test assertions."""
        cls.df_raw, cls.summary = load_raw_dataset(RAW_DATA_PATH, ZIPCODE_POPULATION_PATH)
        cls.df_clean, cls.df_metadata = clean_and_engineer_features(cls.df_raw)
        cls.artifacts = split_and_preprocess_dataset(cls.df_clean, cls.df_metadata)

    def test_raw_data_shape_and_integrity(self):
        """Validates that raw dataset has exactly 7,043 rows and target column."""
        self.assertEqual(len(self.df_raw), 7043)
        self.assertIn("Customer Status", self.df_raw.columns)
        self.assertIn("Population", self.df_raw.columns, "Population from zipcode join should be present")

    def test_target_binarization(self):
        """Ensures target is binary {0, 1} and 'Joined' cohort is treated as retained (0)."""
        y = self.df_clean[TARGET_BINARY_COL]
        self.assertEqual(set(y.unique()), {0, 1})
        
        # Verify Joined customers are 0
        joined_mask = self.df_raw["Customer Status"] == "Joined"
        self.assertTrue((y[joined_mask] == 0).all(), "Joined customers must be mapped to retained (0)")
        
        # Verify Churned customers are 1
        churned_mask = self.df_raw["Customer Status"] == "Churned"
        self.assertTrue((y[churned_mask] == 1).all(), "Churned customers must be mapped to 1")

    def test_target_leakage_absence(self):
        """STRICT GUARDRAIL: Ensures Churn Category and Churn Reason are absent from features."""
        X_train = self.artifacts["X_train"]
        for leakage_col in LEAKAGE_COLS:
            self.assertNotIn(
                leakage_col,
                X_train.columns,
                f"Target leakage hazard: {leakage_col} must NEVER be in X_train!",
            )
        self.assertNotIn(ID_COL, X_train.columns, f"{ID_COL} must not be in training features")

    def test_structural_service_dependencies_imputed(self):
        """Validates that service-dependent missing values are cleanly resolved without nulls."""
        internet_features = [
            "Online Security",
            "Online Backup",
            "Device Protection Plan",
            "Premium Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Streaming Music",
        ]
        no_internet_mask = self.df_clean["Internet Service"] == "No"
        for feat in internet_features:
            vals = self.df_clean.loc[no_internet_mask, feat].unique()
            self.assertTrue(
                "No Internet" in vals or "No" in vals,
                f"Missing values in {feat} for non-internet customers must be categorized",
            )
            self.assertEqual(self.df_clean[feat].isna().sum(), 0, f"No nulls allowed in {feat}")

    def test_engineered_features_presence(self):
        """Checks creation of domain-specific engineered features."""
        engineered = [
            "Num_Internet_Addons",
            "Age_Band",
            "Tenure_Group",
            "Has_Dependents",
            "Has_Referrals",
            "Monthly_to_Total_Revenue_Ratio",
            "Extra_Data_Charge_Ratio",
            "Long_Distance_Charge_Ratio",
        ]
        for feat in engineered:
            self.assertIn(feat, self.df_clean.columns, f"Engineered feature {feat} is missing")

    def test_transformed_splits_null_free(self):
        """Verifies transformed train, val, and test matrices have 0 null values."""
        self.assertEqual(self.artifacts["X_train"].isna().sum().sum(), 0)
        self.assertEqual(self.artifacts["X_val"].isna().sum().sum(), 0)
        self.assertEqual(self.artifacts["X_test"].isna().sum().sum(), 0)

    def test_stratification_consistency(self):
        """Verifies churn rate remains consistent across train, val, and test splits."""
        churn_train = self.artifacts["y_train"].mean()
        churn_val = self.artifacts["y_val"].mean()
        churn_test = self.artifacts["y_test"].mean()
        
        self.assertAlmostEqual(churn_train, churn_val, delta=0.03)
        self.assertAlmostEqual(churn_train, churn_test, delta=0.03)

if __name__ == "__main__":
    unittest.main()
