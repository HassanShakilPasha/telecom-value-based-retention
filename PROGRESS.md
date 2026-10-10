# Project Progress Tracker: Explainable, Value-Based Telecom Churn

**Course / Track**: AI2002 Artificial Intelligence — Track A (R&D)  
**Group**: `F26-11`  
**Last Updated**: October 10, 2026  
**Status**: Phase 1 & Phase 2 Verified & Complete | Ready for Phase 3  

---

## 📊 High-Level Phase Overview & Status Matrix

| Phase | Title | Status | Completion % | Artifacts / Deliverables |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 1** | Context Building & Repository Setup | **Completed** ✅ | 100% | [`README.md`](file:///Users/hassanshakil/Antigravity/AI_Project/README.md), [`AGENTS.md`](file:///Users/hassanshakil/Antigravity/AI_Project/AGENTS.md), Virtual Env |
| **Phase 2** | Data Acquisition & Leak-Free Setup | **Completed** ✅ | 100% | Pipeline scripts, clean datasets, tests, EDA audit report |
| **Phase 3** | Predictive & Cost-Sensitive Modeling | **Pending** ⏳ | 0% | Benchmark models (LR, RF, XGB), Cost curves, Calibrator |
| **Phase 4** | Customer Lifetime Value (CLV) Layer | **Pending** ⏳ | 0% | Historical & Predictive CLV models, Value tiers |
| **Phase 5** | Explainability Layer (XAI via SHAP) | **Pending** ⏳ | 0% | Global SHAP summaries, Local waterfalls, Churn driver maps |
| **Phase 6** | Prescriptive Optimization & Fairness | **Pending** ⏳ | 0% | LP / Knapsack allocator, Demographic parity & fairness audit |

---

## ✅ Phase 1: Context Building & Repository Setup (Completed)

- [x] **Project Scoping**: Formalized Track A (R&D) 4-Layer Machine Learning & Decision Intelligence Architecture.
- [x] **Environment Infrastructure**: Configured Python virtual environment (`venv`) with core data science & ML libraries (`pandas`, `scikit-learn`, `xgboost`, `shap`, `pulp`, `matplotlib`, `seaborn`).
- [x] **Documentation**:
  - [`README.md`](file:///Users/hassanshakil/Antigravity/AI_Project/README.md): Human-oriented project overview, installation instructions, layer definitions, and milestone checklists.
  - [`AGENTS.md`](file:///Users/hassanshakil/Antigravity/AI_Project/AGENTS.md): Authoritative agent execution manual detailing non-negotiable architectural constraints, leakage rules, and mathematical formulations.
  - [`src/utils/config.py`](file:///Users/hassanshakil/Antigravity/AI_Project/src/utils/config.py): Centralized project configuration, paths, column schemas, and logging utilities.

---

## ✅ Phase 2: Data Acquisition & Preprocessing Pipeline (Completed & Verified)

### Accomplished Work
- [x] **Data Ingestion ([`src/data/loader.py`](file:///Users/hassanshakil/Antigravity/AI_Project/src/data/loader.py))**:
  - Ingested primary dataset (`telecom_customer_churn.csv`, 7,043 customer records, 38 columns).
  - Merged California Zip Code population metadata (`telecom_zipcode_population.csv`).
- [x] **Target Definition & Strict Binarization**:
  - Binarized target `Customer Status`: `Churned` $\rightarrow 1$ (1,869 churners, 26.54%), `Stayed` $\rightarrow 0$, `Joined` $\rightarrow 0$ (454 newly acquired cohort retained).
- [x] **Leakage Prevention**:
  - Strictly isolated `Churn Category` and `Churn Reason` alongside `Customer ID` into isolated metadata splits (`metadata_train.csv`, `metadata_val.csv`, `metadata_test.csv`).
  - Zero leakage features present in feature matrices ($X$).
- [x] **Structural Dependency Handling**:
  - Resolved non-random null values across 8 internet service add-ons (`Online Security`, `Device Protection Plan`, etc.) to `"No Internet"`.
  - Replaced missing `Multiple Lines` with `"No Phone"` and null long distance charges with `0.0`.
  - Addressed CSV round-trip string serialization for `Offer` by standardizing null offers to `"No Offer"` (preventing Pandas NaN re-parsing).
- [x] **Feature Engineering**:
  - `Num_Internet_Addons`: Integer count of active security and entertainment services.
  - Financial interaction ratios: `Monthly_to_Total_Revenue_Ratio`, `Extra_Data_Charge_Ratio`, `Long_Distance_Charge_Ratio`, `Refund_to_Total_Charges_Ratio`.
  - Demographic & tenure bands: `Age_Band` (`<30`, `30-50`, `50+`), `Tenure_Group` (`0-12m`, `13-24m`, etc.), `Has_Dependents`, `Has_Referrals`.
- [x] **Leak-Free Transformation Pipeline ([`src/data/preprocessor.py`](file:///Users/hassanshakil/Antigravity/AI_Project/src/data/preprocessor.py))**:
  - Implemented `ColumnTransformer` with 5-fold cross-validated `TargetEncoder` for high-cardinality nominals (`City`, `Zip Code`), `OneHotEncoder` for low-cardinality features, and `StandardScaler` for numeric continuous features.
  - Transformers are fitted strictly on `X_train` and evaluated on validation/test sets.
- [x] **Automated Pipeline Runner ([`scripts/run_phase2_pipeline.py`](file:///Users/hassanshakil/Antigravity/AI_Project/scripts/run_phase2_pipeline.py))**:
  - Produced stratified splits: Train (4,930 rows, 70%), Val (1,056 rows, 15%), Test (1,057 rows, 15%).
  - Serialized pipeline to [`data/processed/preprocessor_pipeline.joblib`](file:///Users/hassanshakil/Antigravity/AI_Project/data/processed/preprocessor_pipeline.joblib).
  - Generated comprehensive exploratory data visualization suite in [`reports/figures/`](file:///Users/hassanshakil/Antigravity/AI_Project/reports/figures/).
  - Generated comprehensive audit documentation in [`reports/phase2_data_audit_report.md`](file:///Users/hassanshakil/Antigravity/AI_Project/reports/phase2_data_audit_report.md).
- [x] **Test Verification**:
  - Passed 57-check comprehensive verification suite and automated unit test suite ([`tests/test_data_pipeline.py`](file:///Users/hassanshakil/Antigravity/AI_Project/tests/test_data_pipeline.py)).

---

## 🔮 Next Steps & Upcoming Phases

### 📌 Phase 3: Layer 1 — Predictive Modeling & Cost-Sensitive Learning
**Objective**: Build, benchmark, and calibrate cost-sensitive churn classifiers that minimize total expected business loss.

1. **Model Architecture Implementation** (`src/models/`):
   - Baseline Classifier: Stratified Dummy / Naive Churn Classifier.
   - Linear Baseline: `LogisticRegression(class_weight='balanced')`.
   - Tree Ensemble: `RandomForestClassifier(n_estimators=200, random_state=42)`.
   - Gradient Boosting: `XGBClassifier(scale_pos_weight=..., eval_metric='aucpr', random_state=42)`.
2. **5-Fold Stratified Cross-Validation**:
   - Evaluate across ROC-AUC, PR-AUC (Average Precision), F1-Macro, Precision, Recall, and Brier Score.
3. **Probability Calibration**:
   - Implement Platt Scaling / Isotonic Calibration to guarantee reliable predicted probabilities $\hat{p}_i$.
4. **Cost-Sensitive Threshold Tuning** (`src/models/cost_sensitive.py`):
   - Shift from symmetric threshold ($\tau=0.5$) to value-based cost curves balancing cost of false negative (customer churn loss) vs false positive (unnecessary incentive cost).
5. **Phase 3 Deliverables**:
   - Training pipeline script: `scripts/run_phase3_modeling.py`.
   - Serialized trained models saved to `models/saved/`.
   - Cross-model comparative report with ROC/PR curves saved to `reports/figures/model_comparison.png`.

---

### 📌 Phase 4: Layer 2 — Customer Lifetime Value (CLV) Layer
**Objective**: Quantify the individual financial value of each customer to separate high-value churn risks from low-value risks.

1. **Historical CLV Calculation** (`src/valuation/clv.py`):
   $$\text{CLV}_{\text{hist}} = \text{Total Charges} - \text{Total Refunds} + \text{Total Extra Data Charges} + \text{Total Long Distance Charges} \equiv \text{Total Revenue}$$
2. **Predictive CLV Modeling**:
   $$\text{CLV}_{\text{pred}} = \text{Monthly Charge} \times \hat{T}_{\text{remaining}}$$
   - Estimate expected remaining customer tenure $\hat{T}_{\text{remaining}}$ based on contract type, survival analysis, or historical hazard rates.
3. **Value Stratification**:
   - Segment customer base into CLV tiers (Platinum, Gold, Silver, Bronze) based on value percentiles.
4. **Phase 4 Deliverables**:
   - Valuation module and unit tests.
   - Value distribution plots and joint Risk-Value matrices saved to `reports/figures/risk_value_matrix.png`.

---

### 📌 Phase 5: Layer 3 — Explainable AI (SHAP)
**Objective**: Uncover root drivers of churn risk globally across the population and locally for individual high-risk customers.

1. **SHAP Integration** (`src/explainability/shap_explainer.py`):
   - Compute `TreeExplainer` values for tree ensembles (XGBoost, Random Forest).
   - Compute `LinearExplainer` values for Logistic Regression.
2. **Global Feature Attribution**:
   - Generate Beeswarm summary plots and mean absolute SHAP feature importance rankings.
3. **Local Customer Explanations**:
   - Generate local force / waterfall plots isolating top 3 positive contributors to churn for flagged subscribers.
4. **Action Mapping Engine**:
   - Map top churn drivers directly to actionable retention categories (e.g., pricing adjustment for high monthly charges, tech onboarding for lack of support, contract discount for month-to-month tenure).
5. **Validation Against Ground Truth**:
   - Correlate SHAP top drivers with ground-truth `Churn Reason` metadata in churned cohort to validate model explanations.
6. **Phase 5 Deliverables**:
   - SHAP explanation plots saved in `reports/figures/shap_summary.png`.
   - Individual waterfall explanation cards for retention agents.

---

### 📌 Phase 6: Layer 4 — Prescriptive Optimization & Algorithmic Fairness
**Objective**: Optimal budget allocation for targeted retention campaigns subject to business constraints, coupled with demographic fairness guarantees.

1. **Constrained Retention Optimizer** (`src/prescription/optimizer.py`):
   $$\max_{\{x_i \in \{0, 1\}\}} \sum_{i=1}^N x_i \cdot \Big( \hat{p}_i \times \text{CLV}_i \times \delta - c_i \Big) \quad \text{s.t.} \quad \sum_{i=1}^N x_i \cdot c_i \le B$$
   - Solve using Mixed-Integer Linear Programming (`pulp` / Knapsack greedy solver) under varying budget scenarios ($B \in [\$5,000, \$50,000]$).
2. **Algorithmic Fairness Audit** (`src/prescription/fairness.py`):
   - Audit decisions across protected/sensitive demographics: `Age_Band`, `Gender`, `Married`, `Has_Dependents`.
   - Metrics: Equal Opportunity Difference ($|TPR_A - TPR_B|$), Disparate Impact Ratio, and False Positive Parity.
   - Enforce parity thresholds to ensure equitable retention campaign distribution.
3. **Phase 6 Deliverables**:
   - Optimization simulation engine and budget curve plots.
   - Comprehensive Fairness Audit Report in `reports/fairness_audit_report.md`.
   - Final End-to-End Executive Summary and Presentation materials.

---

## 📁 Verified Artifacts Summary

| Artifact Name | Path | Description |
| :--- | :--- | :--- |
| **Pipeline Runner** | [`scripts/run_phase2_pipeline.py`](file:///Users/hassanshakil/Antigravity/AI_Project/scripts/run_phase2_pipeline.py) | Full Phase 2 execution script |
| **Data Loader** | [`src/data/loader.py`](file:///Users/hassanshakil/Antigravity/AI_Project/src/data/loader.py) | Raw data ingestion & zipcode merge |
| **Preprocessor** | [`src/data/preprocessor.py`](file:///Users/hassanshakil/Antigravity/AI_Project/src/data/preprocessor.py) | Dependency imputation, hybrid encoding & feature engineering |
| **Config** | [`src/utils/config.py`](file:///Users/hassanshakil/Antigravity/AI_Project/src/utils/config.py) | Global constants, paths, column sets |
| **Unit Tests** | [`tests/test_data_pipeline.py`](file:///Users/hassanshakil/Antigravity/AI_Project/tests/test_data_pipeline.py) | 7 automated unit tests for data pipeline integrity |
| **Clean Dataset** | `data/processed/telecom_churn_cleaned.csv` | Cleaned full dataset (7,043 rows, 45 columns) |
| **Encoded Train/Val/Test**| `data/processed/*_encoded.csv` | Fully encoded splits (45 feature columns + Churn) |
| **Raw Splits** | `data/processed/*_raw.csv` | Un-encoded feature splits for tree-based algorithms |
| **Metadata Splits** | `data/processed/metadata_*.csv` | Isolated IDs, targets, and leakage fields |
| **Trained Pipeline** | `data/processed/preprocessor_pipeline.joblib` | Serialized `ColumnTransformer` (StandardScaler + TargetEncoder + OneHotEncoder) |
| **Audit Report** | [`reports/phase2_data_audit_report.md`](file:///Users/hassanshakil/Antigravity/AI_Project/reports/phase2_data_audit_report.md) | In-depth data validation and statistical summary |
