# AGENTS.md — Autonomous Agent Operating Manual & Repository Context

> **Audience**: AI Coding Agents, Copilots, and Automated Runners working on this codebase.  
> **Repository**: `AI_Project` (`/Users/hassanshakil/Antigravity/AI_Project`)  
> **Course / Group**: AI2002 Artificial Intelligence — Track A (R&D) | Group `F26-11`  
> **Core Theme**: Explainable, Value-Based Customer Churn Prediction and Prescription in Telecom.

---

## 1. Project Mission & High-Level Objective

Traditional churn prediction models optimize solely for statistical classification metrics (Accuracy, ROC-AUC) without considering the **economic value** of individual customers or providing **prescriptive, budget-optimized interventions**.

This project implements an end-to-end **4-Layer Machine Learning & Decision Intelligence Framework**:
1. **Layer 1: Predictive Modeling** — Cost-sensitive churn classification (Logistic Regression, Random Forest, XGBoost) with CLV-weighted loss and optimal threshold calibration.
2. **Layer 2: Customer Lifetime Value (CLV)** — Estimation of customer financial value using historical billing metrics and predictive tenure modeling.
3. **Layer 3: Explainability (XAI via SHAP)** — Global and local feature attributions that reveal root causes of churn for individual flagged customers.
4. **Layer 4: Prescriptive Retention & Optimization** — Budget-constrained knapsack/linear programming assigning tailored retention actions to maximize net retained value, paired with demographic fairness audits.

---

## 2. Non-Negotiable Architectural Constraints & Guardrails

When generating, modifying, or executing code in this repository, agents **MUST** enforce the following rules:

### ⚠️ Constraint 1: Strict Target Leakage Prevention
- **Columns `Churn Category` and `Churn Reason` must NEVER enter the feature matrix ($X$) during model training or feature selection.**
- These fields are recorded *exclusively after a customer churns*. Using them guarantees artificial 100% accuracy and complete operational failure.
- *Permitted Use*: They may be used in post-hoc validation to compare whether SHAP-identified top drivers correlate with the actual self-reported exit reasons.

### ⚠️ Constraint 2: Target Variable Definition & Binarization
- The target column in `telecom_customer_churn.csv` is `Customer Status`.
- Class breakdown:
  - `Churned` $\rightarrow 1$ (Positive class, minority ~26.5%)
  - `Stayed` $\rightarrow 0$ (Negative class ~67.0%)
  - `Joined` $\rightarrow 0$ (Negative class ~6.4% — newly acquired subscribers during Q2 2022; documented explicitly as retained).
- **Binary Target**: $y = 1$ if `Customer Status == 'Churned'`, else $0$.

### ⚠️ Constraint 3: Structural Missing Values vs. Data Corruption
- Many missing values in `telecom_customer_churn.csv` are **structural dependencies**, not missing at random:
  - If `Internet Service == 'No'`, then: `Internet Type`, `Online Security`, `Online Backup`, `Device Protection Plan`, `Premium Tech Support`, `Streaming TV`, `Streaming Movies`, `Streaming Music`, `Unlimited Data` are null/empty.
  - If `Phone Service == 'No'`, then: `Multiple Lines` is null/empty, and `Avg Monthly Long Distance Charges` is $0$.
- **Do NOT blindly drop rows with missing values or use unconditional mean/median imputation.**
- Replace missing service flags with explicit categories such as `"No Internet"` or encode them logically.

### ⚠️ Constraint 4: Data Splitting & Leakage-Free Preprocessing
- Always perform stratified splitting (`StratifiedKFold` or `train_test_split(stratify=y, random_state=42)`).
- **All encoders (OneHotEncoder, TargetEncoder) and scalers (StandardScaler, RobustScaler) must be fit ONLY on the training fold** and transformed on the test/validation fold.
- Never fit transformations on the entire dataset prior to splitting.

---

## 3. Mathematical Formulations & Layer Specifications

### Layer 1: Cost-Sensitive Classification & Thresholding
- Standard probability threshold $\tau = 0.5$ assumes symmetric error costs ($C_{\text{FP}} = C_{\text{FN}}$). In churn retention:
  - $C_{\text{FN}}$ (False Negative: missing a true churner) = Loss of customer CLV.
  - $C_{\text{FP}}$ (False Positive: offering an incentive to a loyal customer) = Cost of unnecessary retention offer ($C_{\text{offer}}$).
- Agents must implement a value-based cost threshold $\tau_i^*$ tailored to customer $i$:
  $$\tau_i^* = \frac{C_{\text{offer}}}{P(\text{retention succeeds}) \times \text{CLV}_i}$$
- Benchmarked Algorithms:
  1. `LogisticRegression(class_weight='balanced')`
  2. `RandomForestClassifier(n_estimators=200, random_state=42)`
  3. `XGBClassifier(scale_pos_weight=..., eval_metric='aucpr', random_state=42)`

### Layer 2: Customer Lifetime Value (CLV)
Two models must be calculated:
1. **Historical CLV**:
   $$\text{CLV}_{\text{hist}} = \text{Total Charges} - \text{Total Refunds} + \text{Total Extra Data Charges} + \text{Total Long Distance Charges} \equiv \text{Total Revenue}$$
2. **Predictive CLV**:
   $$\text{CLV}_{\text{pred}} = \text{Monthly Charge} \times \hat{T}_{\text{remaining}}$$
   where $\hat{T}_{\text{remaining}}$ is estimated via Kaplan-Meier survival curves or expected remaining contract months.

### Layer 3: Explainable AI (SHAP)
- Use `shap.TreeExplainer` for XGBoost and Random Forest.
- Use `shap.LinearExplainer` for Logistic Regression.
- For every customer flagged with $P(\text{Churn}) > \tau^*$, extract the top 3 positive contributors to churn risk ($\phi_{i, j} > 0$).
- Map top drivers to prescriptive action categories:
  - High Monthly Charge $\rightarrow$ Pricing / Plan Restructuring
  - High Long Distance / Extra Data Charges $\rightarrow$ Unlimited Bundle Offer
  - Tech / Support Dissatisfaction $\rightarrow$ Dedicated Concierge Support
  - Month-to-Month Contract $\rightarrow$ Annual Contract Discount Incentive

### Layer 4: Prescriptive Retention Optimization & Fairness
- **Risk $\times$ Value Baseline Grid**:
  - Quadrant 1: High Churn Risk ($>0.5$), High CLV ($> \text{Median}$) $\rightarrow$ Tier 1 Intervention.
  - Quadrant 2: High Churn Risk ($>0.5$), Low CLV ($\le \text{Median}$) $\rightarrow$ Low-Cost Digital Campaign.
  - Quadrant 3: Low Churn Risk ($\le 0.5$), High CLV ($> \text{Median}$) $\rightarrow$ Loyalty Rewards.
  - Quadrant 4: Low Churn Risk ($\le 0.5$), Low CLV ($\le \text{Median}$) $\rightarrow$ No Action.
- **Constrained Optimization Formulation**:
  $$\max_{\{x_i \in \{0, 1\}\}} \sum_{i=1}^N x_i \cdot \Big( \hat{p}_i \times \text{CLV}_i \times \delta - c_i \Big) \quad \text{s.t.} \quad \sum_{i=1}^N x_i \cdot c_i \le B$$
  where $\hat{p}_i = P(\text{Churn}_i)$, $\delta$ is the retention intervention lift/success rate (e.g., $0.35$), $c_i$ is intervention cost, and $B$ is the total retention budget.
- **Fairness Audit Requirements**:
  - Subgroups to audit: `Age Band` ($<30$, $30-50$, $50+$), `Gender` (Male vs Female), `Married` (Yes vs No), `Number of Dependents` ($0$ vs $>0$).
  - Metrics: Equal Opportunity Difference ($|TPR_A - TPR_B|$), False Positive Parity, Disparate Impact Ratio.

---

## 4. Dataset Schema Reference

Data source: `Telecom+Customer+Churn/`

| Field Name | Type | Key Details & Handling |
| :--- | :--- | :--- |
| `Customer ID` | String | Unique primary key. Drop from training feature matrix. |
| `Gender` | String | Male, Female. Protected attribute for fairness audit. |
| `Age` | Int | Customer age in years. Protected attribute for fairness audit. |
| `Married` | String | Yes, No. |
| `Number of Dependents` | Int | Dependents living with customer. |
| `City` | String | Primary California city. |
| `Zip Code` | Int/Str | California zip code (joinable with `telecom_zipcode_population.csv`). |
| `Latitude`, `Longitude`| Float | Spatial coordinates for geospatial feature engineering. |
| `Number of Referrals` | Int | Count of referrals made to date. |
| `Tenure in Months` | Int | Total tenure with company. Key predictor of loyalty. |
| `Offer` | String | Last accepted marketing offer (None, Offer A–E). |
| `Phone Service` | String | Yes, No. Gate for phone-related services. |
| `Avg Monthly Long Distance Charges` | Float | 0 if Phone Service is No. |
| `Multiple Lines` | String | Yes, No (No if Phone Service is No). |
| `Internet Service` | String | Yes, No. Gate for internet-related services. |
| `Internet Type` | String | DSL, Fiber Optic, Cable, None. |
| `Avg Monthly GB Download`| Int | Download volume (0 if Internet Service is No). |
| `Online Security` | String | Service add-on (Yes, No, No Internet). |
| `Online Backup` | String | Service add-on (Yes, No, No Internet). |
| `Device Protection Plan`| String | Service add-on (Yes, No, No Internet). |
| `Premium Tech Support` | String | Service add-on (Yes, No, No Internet). |
| `Streaming TV` | String | Service add-on (Yes, No, No Internet). |
| `Streaming Movies` | String | Service add-on (Yes, No, No Internet). |
| `Streaming Music` | String | Service add-on (Yes, No, No Internet). |
| `Unlimited Data` | String | Add-on for unlimited data (Yes, No). |
| `Contract` | String | Month-to-Month, One Year, Two Year. Very strong churn signal. |
| `Paperless Billing` | String | Yes, No. |
| `Payment Method` | String | Bank Withdrawal, Credit Card, Mailed Check. |
| `Monthly Charge` | Float | Total recurring monthly charge. |
| `Total Charges` | Float | Cumulative total billed charges. |
| `Total Refunds` | Float | Cumulative refunds provided. |
| `Total Extra Data Charges` | Float | Excess data charges. |
| `Total Long Distance Charges` | Float | Excess long-distance charges. |
| `Total Revenue` | Float | Net company revenue from customer. |
| `Customer Status` | String | **Target**: Churned, Stayed, Joined. |
| `Churn Category` | String | **LEAKAGE**: Strictly omit from training! |
| `Churn Reason` | String | **LEAKAGE**: Strictly omit from training! |

---

## 5. Coding Standards & Recommended Project Structure

When creating new modules or pipelines, adhere to this modular structure:

```
src/
├── data/
│   ├── loader.py             # Robust data ingestion & joins with zipcode population
│   └── preprocessor.py       # Imputation of service dependencies, encoding, scaling
├── models/
│   ├── train.py              # Cross-validated training of LR, RF, XGBoost
│   ├── evaluate.py           # Comprehensive metric computation (ROC, PR, Cost curve)
│   └── cost_sensitive.py     # Custom threshold optimization and weighted loss
├── valuation/
│   └── clv.py                # Historical and predictive CLV calculations
├── explainability/
│   └── shap_explainer.py     # TreeSHAP calculation, summary plots, local waterfalls
├── prescription/
│   ├── optimizer.py          # Knapsack / Linear programming retention allocator
│   └── fairness.py           # Subgroup disparity and fairness auditing
└── utils/
    └── config.py             # Global constants, filepaths, random seed (42)
```

### Code Style Guidelines:
- Write modular, testable functions with Python type annotations (`typing`).
- Ensure reproducibility by setting random seeds:
  ```python
  import numpy as np, random, os
  SEED = 42
  np.random.seed(SEED)
  random.seed(SEED)
  ```
- Use `logging` instead of unformatted `print()` statements for pipeline execution logs.
- Save trained artifacts and evaluation tables to `models/saved/` and `reports/figures/`.

---

## 6. CLI & Execution Cheat Sheet

```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core data science stack
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn scipy pulp

# Run pipeline test
python3 -m unittest discover tests/
```

---

*This document is maintained as the authoritative instruction manual for all AI agents assisting in Group F26-11's AI2002 Project.*
