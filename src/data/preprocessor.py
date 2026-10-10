"""
Data Cleaning, Feature Engineering, and Leakage-Free Preprocessing Module.
Enforces structural service dependency handling, target binarization, feature engineering,
and stratified train/val/test splits with encoder fitting exclusively on the training fold.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer

from src.utils.config import (
    PROCESSED_DATA_DIR,
    SEED,
    TARGET_RAW_COL,
    TARGET_BINARY_COL,
    ID_COL,
    LEAKAGE_COLS,
    INTERNET_DEPENDENT_COLS,
    PHONE_DEPENDENT_COLS,
    setup_logging,
)

logger = setup_logging()

def clean_and_engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans raw telecom data, resolves structural dependencies, and engineers new features.
    
    Returns:
        df_clean (pd.DataFrame): Cleaned and enriched dataset with target and features.
        df_metadata (pd.DataFrame): Post-churn fields and IDs isolated for post-hoc validation.
    """
    logger.info("Starting cleaning and feature engineering...")
    df = df.copy()

    # 1. Target Binarization
    # 'Churned' -> 1; 'Stayed' -> 0; 'Joined' -> 0 (retained newly acquired cohort)
    logger.info(f"Binarizing target '{TARGET_RAW_COL}'...")
    df[TARGET_BINARY_COL] = (df[TARGET_RAW_COL] == "Churned").astype(int)
    
    # 2. Extract and Isolate Post-Churn & Identity Metadata (to strictly avoid leakage)
    metadata_cols = [ID_COL, TARGET_RAW_COL] + [c for c in LEAKAGE_COLS if c in df.columns]
    # Retain key financial metrics in metadata for Layer 2 & 4 calculations
    for fin_col in ["Total Revenue", "Monthly Charge", "Total Charges", "Total Refunds"]:
        if fin_col in df.columns and fin_col not in metadata_cols:
            metadata_cols.append(fin_col)
            
    df_metadata = df[metadata_cols].copy()
    logger.info(f"Isolated metadata & post-churn leakage fields: {metadata_cols}")

    # 3. Handle Structural Missing Dependencies
    # Internet Service Dependencies
    has_internet = df["Internet Service"] == "Yes"
    
    # Internet Type: if No internet, assign 'No Internet'
    df["Internet Type"] = df["Internet Type"].fillna("No Internet")
    # Avg Monthly GB Download: if No internet, 0
    df["Avg Monthly GB Download"] = df["Avg Monthly GB Download"].fillna(0)
    
    # Internet add-ons
    internet_addons = [
        "Online Security",
        "Online Backup",
        "Device Protection Plan",
        "Premium Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Streaming Music",
        "Unlimited Data",
    ]
    for addon in internet_addons:
        if addon in df.columns:
            df[addon] = df[addon].fillna("No Internet")

    # Phone Service Dependencies
    has_phone = df["Phone Service"] == "Yes"
    if "Multiple Lines" in df.columns:
        df["Multiple Lines"] = df["Multiple Lines"].fillna("No Phone")
    if "Avg Monthly Long Distance Charges" in df.columns:
        df["Avg Monthly Long Distance Charges"] = df["Avg Monthly Long Distance Charges"].fillna(0.0)

    # Offer handling (if None / null, set to 'No Offer')
    # NOTE: Do NOT use the string "None" — pandas re-parses it as NaN on CSV reload.
    if "Offer" in df.columns:
        df["Offer"] = df["Offer"].fillna("No Offer")

    # California Population (if null from zipcode join, impute with median)
    if "Population" in df.columns:
        df["Population"] = df["Population"].fillna(df["Population"].median())
    if "Zip Code" in df.columns:
        df["Zip Code"] = df["Zip Code"].astype(str)

    # 4. Feature Engineering
    logger.info("Engineering domain features (add-on count, charge ratios, demographic bands)...")
    
    # Service Count: Total internet security/entertainment services subscribed
    service_cols = [
        "Online Security",
        "Online Backup",
        "Device Protection Plan",
        "Premium Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Streaming Music",
    ]
    df["Num_Internet_Addons"] = df[service_cols].apply(
        lambda row: sum(row == "Yes"), axis=1
    )

    # Demographic bands for Layer 4 Fairness auditing
    df["Age_Band"] = pd.cut(
        df["Age"],
        bins=[0, 30, 50, 120],
        labels=["<30", "30-50", "50+"],
        right=False,
    ).astype(str)

    df["Tenure_Group"] = pd.cut(
        df["Tenure in Months"],
        bins=[0, 12, 24, 48, 120],
        labels=["0-12m", "13-24m", "25-48m", "49+m"],
        right=False,
    ).astype(str)

    df["Has_Dependents"] = (df["Number of Dependents"] > 0).astype(int)
    df["Has_Referrals"] = (df["Number of Referrals"] > 0).astype(int)

    # Financial charge interaction ratios
    df["Monthly_to_Total_Revenue_Ratio"] = df["Monthly Charge"] / (df["Total Revenue"] + 1.0)
    df["Extra_Data_Charge_Ratio"] = df["Total Extra Data Charges"] / (df["Total Charges"] + 1.0)
    df["Long_Distance_Charge_Ratio"] = df["Total Long Distance Charges"] / (df["Total Charges"] + 1.0)
    df["Refund_to_Total_Charges_Ratio"] = df["Total Refunds"] / (df["Total Charges"] + 1.0)

    logger.info("Feature engineering complete.")
    return df, df_metadata

def build_preprocessor_pipeline(
    X_train: pd.DataFrame,
    random_state: int = SEED,
) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    """
    Constructs a ColumnTransformer with:
    - TargetEncoder for high-cardinality nominals (e.g. City, Zip Code)
    - OneHotEncoder for low-cardinality categoricals
    - StandardScaler for continuous features.
    Derived exclusively from X_train columns.
    """
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    high_card_features = [c for c in categorical_features if X_train[c].nunique() > 15]
    low_card_features = [c for c in categorical_features if X_train[c].nunique() <= 15]

    logger.info(
        f"Identified {len(numeric_features)} numeric, {len(high_card_features)} high-cardinality, "
        f"and {len(low_card_features)} low-cardinality categorical features."
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "high_card",
                TargetEncoder(smooth="auto", cv=cv),
                high_card_features,
            ),
            (
                "low_card",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                low_card_features,
            ),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, high_card_features, low_card_features

def split_and_preprocess_dataset(
    df: pd.DataFrame,
    df_metadata: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = SEED,
) -> Dict[str, Any]:
    """
    Executes a stratified, leak-free train/val/test split and fits transformers
    strictly on the training fold.
    
    Returns:
        dict: Preprocessed dataframes, feature names, transformers, and metadata splits.
    """
    logger.info("Preparing feature matrix X and target y...")
    
    # Exclude ID, target, and leakages from features
    drop_from_X = [ID_COL, TARGET_RAW_COL, TARGET_BINARY_COL] + [
        c for c in LEAKAGE_COLS if c in df.columns
    ]
    X = df.drop(columns=[col for col in drop_from_X if col in df.columns])
    y = df[TARGET_BINARY_COL]

    logger.info(f"Features dimension: {X.shape}, Target dimension: {y.shape}")

    # Stratified Train / (Val + Test) split
    temp_size = test_size + val_size
    X_train_raw, X_temp_raw, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, df.index, test_size=temp_size, random_state=random_state, stratify=y
    )

    # Split temp into Val and Test
    relative_test_size = test_size / temp_size
    X_val_raw, X_test_raw, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp_raw, y_temp, idx_temp, test_size=relative_test_size, random_state=random_state, stratify=y_temp
    )

    logger.info(
        f"Splits created: Train={X_train_raw.shape[0]} ({y_train.mean():.1%} churn), "
        f"Val={X_val_raw.shape[0]} ({y_val.mean():.1%} churn), "
        f"Test={X_test_raw.shape[0]} ({y_test.mean():.1%} churn)"
    )

    # Build and fit ColumnTransformer strictly on Training Fold
    preprocessor, num_cols, high_card_cols, low_card_cols = build_preprocessor_pipeline(X_train_raw, random_state=random_state)
    logger.info("Fitting ColumnTransformer exclusively on X_train and y_train...")
    preprocessor.fit(X_train_raw, y_train)

    # Transform all splits
    X_train_trans = preprocessor.transform(X_train_raw)
    X_val_trans = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # Extract transformed feature names
    high_names = preprocessor.named_transformers_["high_card"].get_feature_names_out(high_card_cols).tolist()
    low_encoder = preprocessor.named_transformers_["low_card"]
    low_names = low_encoder.get_feature_names_out(low_card_cols).tolist()
    feature_names = num_cols + high_names + low_names

    # Convert back to clean DataFrames with feature names
    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names, index=idx_train)
    X_val_df = pd.DataFrame(X_val_trans, columns=feature_names, index=idx_val)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=idx_test)

    # Extract metadata splits
    meta_train = df_metadata.loc[idx_train]
    meta_val = df_metadata.loc[idx_val]
    meta_test = df_metadata.loc[idx_test]

    return {
        "X_train": X_train_df,
        "X_val": X_val_df,
        "X_test": X_test_df,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "meta_train": meta_train,
        "meta_val": meta_val,
        "meta_test": meta_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numeric_features": num_cols,
        "categorical_features": high_card_cols + low_card_cols,
        "high_card_features": high_card_cols,
        "low_card_features": low_card_cols,
    }

def save_processed_artifacts(
    artifacts: Dict[str, Any],
    df_clean: pd.DataFrame,
    output_dir: Path = PROCESSED_DATA_DIR,
):
    """
    Saves processed datasets, metadata, and the fitted pipeline transformer to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving processed artifacts to {output_dir}...")

    # Save complete clean dataset
    # NOTE: Churn Category/Reason are post-event fields; NaN for non-churned customers is expected.
    df_clean.to_csv(output_dir / "telecom_churn_cleaned.csv", index=False)
    
    # Save transformed training, validation, and test datasets
    train_df = artifacts["X_train"].copy()
    train_df[TARGET_BINARY_COL] = artifacts["y_train"].values
    train_df.to_csv(output_dir / "train_encoded.csv", index=False)

    val_df = artifacts["X_val"].copy()
    val_df[TARGET_BINARY_COL] = artifacts["y_val"].values
    val_df.to_csv(output_dir / "val_encoded.csv", index=False)

    test_df = artifacts["X_test"].copy()
    test_df[TARGET_BINARY_COL] = artifacts["y_test"].values
    test_df.to_csv(output_dir / "test_encoded.csv", index=False)

    # Save raw non-encoded splits (useful for Tree-based models like CatBoost/XGBoost native categoricals)
    artifacts["X_train_raw"].assign(Churn=artifacts["y_train"]).to_csv(
        output_dir / "train_raw.csv", index=False
    )
    artifacts["X_val_raw"].assign(Churn=artifacts["y_val"]).to_csv(
        output_dir / "val_raw.csv", index=False
    )
    artifacts["X_test_raw"].assign(Churn=artifacts["y_test"]).to_csv(
        output_dir / "test_raw.csv", index=False
    )

    # Save metadata splits
    artifacts["meta_train"].to_csv(output_dir / "metadata_train.csv", index=False)
    artifacts["meta_val"].to_csv(output_dir / "metadata_val.csv", index=False)
    artifacts["meta_test"].to_csv(output_dir / "metadata_test.csv", index=False)

    # Save fitted ColumnTransformer
    joblib.dump(artifacts["preprocessor"], output_dir / "preprocessor_pipeline.joblib")
    
    logger.info("All processed artifacts successfully serialized to disk.")
