# Phase 2 Data Quality, Acquisition & Setup Audit Report

**Course**: AI2002 – Artificial Intelligence | Track A: Research & Development (R&D)  
**Group ID**: `F26-11`  
**Date**: October 2026  
**Status**: COMPLETED & VERIFIED

---

## 1. Data Ingestion & Schema Summary

| Property | Value |
| :--- | :--- |
| **Total Customers Ingested** | `7,043` |
| **Raw Columns Count** | `39` |
| **California Zip Code Populations Joined** | Yes (Source: `telecom_zipcode_population.csv`) |
| **Engineered Features Added** | `Num_Internet_Addons`, `Age_Band`, `Tenure_Group`, `Has_Dependents`, `Has_Referrals`, `Monthly_to_Total_Revenue_Ratio`, `Extra_Data_Charge_Ratio`, `Long_Distance_Charge_Ratio`, `Refund_to_Total_Charges_Ratio` |
| **Total Transformed Features ($D$)** | `84` (Numerical: 22, Target-Encoded High-Cardinality: 2, One-Hot Encoded Low-Cardinality: 60) |

---

## 2. Target Variable & Class Distribution

- **Raw Breakdown**:
  - `Stayed`: `4,720` (67.0%)
  - `Churned`: `1,869` (26.5%)
  - `Joined`: `454` (6.4%)
- **Binarized Churn Target ($y$)**:
  - Positive Class ($y = 1$, Churned): `1,869` (26.5%)
  - Negative Class ($y = 0$, Retained): `5,174` (73.5%)
  - *Note on 'Joined' Cohort*: Handled strictly as retained ($0$) because these subscribers joined in Q2 2022 and have not exited.

---

## 3. Data Integrity & Leakage Prevention Guardrail Audit

| Guardrail Test | Expected Rule | Audit Result | Status |
| :--- | :--- | :--- | :--- |
| **Target Leakage (`Churn Category`)** | Excluded from $X$ | Absent in $X_{train}, X_{val}, X_{test}$ | ✅ PASS |
| **Target Leakage (`Churn Reason`)** | Excluded from $X$ | Absent in $X_{train}, X_{val}, X_{test}$ | ✅ PASS |
| **Primary Key Leakage (`Customer ID`)** | Excluded from $X$ | Absent in $X_{train}, X_{val}, X_{test}$ | ✅ PASS |
| **Missing Values in Transformed Splits** | 0 null cells | Train: `0`, Val: `0`, Test: `0` | ✅ PASS |
| **Structural Dependency Imputation** | Logical imputation (e.g., `'No Internet'`) | All service dependencies resolved without dropping rows | ✅ PASS |
| **Data Leakage in Transformations** | Fit on Train fold ONLY | `preprocessor` fitted exclusively on `X_train` | ✅ PASS |

---

## 4. Stratified Dataset Partitions

| Partition | Total Records | Non-Churn ($y=0$) | Churn ($y=1$) | Churn Proportion |
| :--- | :--- | :--- | :--- | :--- |
| **Training (70%)** | `4,930` | `3,622` | `1,308` | `26.53%` |
| **Validation (15%)** | `1,056` | `776` | `280` | `26.52%` |
| **Test (15%)** | `1,057` | `776` | `281` | `26.58%` |
| **Total** | `7,043` | `5,174` | `1,869` | `26.54%` |

---

## 5. Serialized Artifacts Checklist

All artifacts are persisted under `data/processed/` and `reports/figures/`:
- `data/processed/telecom_churn_cleaned.csv` — Full cleaned dataset with metadata and engineered features.
- `data/processed/train_encoded.csv` — Encoded and scaled training matrix.
- `data/processed/val_encoded.csv` — Encoded and scaled validation matrix.
- `data/processed/test_encoded.csv` — Encoded and scaled test evaluation matrix.
- `data/processed/train_raw.csv`, `val_raw.csv`, `test_raw.csv` — Un-encoded splits for native tree architectures.
- `data/processed/metadata_train.csv`, `metadata_val.csv`, `metadata_test.csv` — Customer IDs, revenue, and ground-truth exit reasons for downstream XAI and Prescription.
- `data/processed/preprocessor_pipeline.joblib` — Serialized Scikit-Learn `ColumnTransformer`.
- `reports/figures/target_distribution.png` — Target class distribution plot.
- `reports/figures/churn_by_contract.png` — Churn rate breakdown by contract type.
- `reports/figures/churn_by_age_and_dependents.png` — Churn rate across demographic fairness subgroups.
- `reports/figures/correlation_matrix.png` — Correlation matrix of continuous features.
