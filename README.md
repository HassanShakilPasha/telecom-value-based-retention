# From Prediction to Prescription: An Explainable AI Framework for Value-Based Customer Retention in the Telecom Sector

[![Course](https://img.shields.io/badge/Course-AI2002%20Artificial%20Intelligence-blue.svg)](https://github.com/HassanShakilPasha/telecom-value-based-retention)
[![Track](https://img.shields.io/badge/Track-A%3A%20Research%20%26%20Development-success.svg)](https://github.com/HassanShakilPasha/telecom-value-based-retention)
[![Group](https://img.shields.io/badge/Group%20ID-F26--11-orange.svg)](https://github.com/HassanShakilPasha/telecom-value-based-retention)
[![Dataset](https://img.shields.io/badge/Dataset-Maven%20Analytics%20Telecom-yellow.svg)](https://mavenanalytics.io/data-playground?search=churn)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 👥 Project Information & Team Members

- **Course**: AI2002 – Artificial Intelligence
- **Track**: Track A: Research & Development (R&D)
- **Group ID**: `F26-11`

### 🧑‍💻 Team Members
| Name | Roll Number | Role / Primary Contributions |
| :--- | :--- | :--- |
| **Saqib Qazi** | `24L-2506` | Predictive Modeling, Imbalance & Cost-Sensitive Learning |
| **Hassan Shakil** | `24L-2521` | CLV Valuation Architecture, Prescription & Optimization |
| **Ammara Saleem** | `24L-2571` | Explainable AI (SHAP), Fairness Auditing & Dashboarding |

---

## 📌 Executive Summary & Problem Statement

Customer churn is one of the costliest operational problems in the telecommunications sector. Acquiring a replacement subscriber costs between **5× to 25× more** than retaining an existing one. 

While conventional machine learning literature extensively treats churn prediction as a binary classification problem ($y \in \{0, 1\}$) evaluated primarily on standard accuracy or AUC-ROC, standard industry deployments encounter **three critical practical gaps**:

1. **No Value-Weighting (Indifference to Customer Worth)**: A churn-prone subscriber paying \$15/month is treated with the same urgency as a high-margin business subscriber generating \$120/month, even though the business risk and economic impact diverge sharply.
2. **No Individual-Level Explanations (Black-Box Predictions)**: Global feature importance metrics (e.g., Gini index or permutation importance) describe the overall dataset trend, but fail to tell a front-line retention specialist *why a specific customer is leaving* (e.g., competitor optical fiber pricing vs. poor customer support).
3. **No Prescriptive Guidance (Scores $\neq$ Interventions)**: Predicting high churn risk does not instruct the retention team on *what action to take* or *who to prioritize under finite marketing budgets*.

### 🔬 Core Research Question
> **"Can a unified four-layer framework that integrates churn prediction, Customer Lifetime Value (CLV) estimation, and SHAP-based explainability produce a more cost-effective and actionable retention strategy than traditional standalone churn classification?"**

---

## 🏗️ The 4-Layer Architectural Framework

The project is structured into four interconnected modules:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       1. PREDICTIVE MODELING LAYER                          │
│  - Preprocessing & Service-Dependency Null Imputation                       │
│  - Models: Logistic Regression, Random Forest, XGBoost                     │
│  - Cost-Sensitive Learning & CLV-Weighted Classification Loss                │
│  - Metric Suite: Recall, Precision, PR-AUC, ROC-AUC, Cost Curve             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ P(Churn)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          2. VALUATION LAYER (CLV)                           │
│  - Historical CLV (Billing, Service Add-ons, Adjustments, Refunds)          │
│  - Predictive CLV (Monthly Bill × Expected Remaining Tenure via Survival)   │
│  - Comparative evaluation & assignment of customer economic worth           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ CLV Estimates
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     3. EXPLAINABILITY LAYER (SHAP XAI)                      │
│  - Global Explanations (Summary Beeswarm, Mean |SHAP| Rankings)              │
│  - Local Explanations (Waterfall & Force Plots for Flagged Subscribers)     │
│  - Root-Cause Categorization (Pricing, Competitor, Support, Contract)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Top Churn Drivers
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         4. PRESCRIPTIVE ACTION LAYER                        │
│  - Value-Risk Matrix Prioritization (High Risk × High Value)                │
│  - Linear Programming / Knapsack Optimization under Budget Constraints       │
│  - Targeted Intervention Assignment (Discount, Upgrade, VIP Retention)      │
│  - Demographic Fairness & Subgroup Disparity Audit                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Profile

The dataset utilized is the **Maven Analytics Telecom Customer Churn Dataset** representing 7,043 residential and business customers of a California-based telecommunications provider.

- **Sample Size**: 7,043 customer records
- **Feature Dimension**: 38 attributes + California Zipcode Population mappings
- **Target Distribution**:
  - `Stayed`: 4,720 customers (**67.0%**)
  - `Churned`: 1,869 customers (**26.5%**)
  - `Joined`: 454 customers (**6.4%**) — *Newly acquired subscribers during Q2 2022. Handled as non-churned (retained).*
  - **Effective Binary Class Imbalance**: ~26.5% Churn vs. ~73.5% Retained.

### Feature Domain Groupings
1. **Demographics**: `Gender`, `Age`, `Married`, `Number of Dependents`, `City`, `Zip Code`, `Latitude`, `Longitude`.
2. **Account & Contract**: `Tenure in Months`, `Contract` (Month-to-Month, One Year, Two Year), `Paperless Billing`, `Payment Method`, `Offer` (Offers A–E, None).
3. **Telephony Services**: `Phone Service`, `Multiple Lines`, `Avg Monthly Long Distance Charges`.
4. **Internet & Streaming Add-ons**: `Internet Service`, `Internet Type` (DSL, Fiber Optic, Cable), `Avg Monthly GB Download`, `Online Security`, `Online Backup`, `Device Protection Plan`, `Premium Tech Support`, `Streaming TV`, `Streaming Movies`, `Streaming Music`, `Unlimited Data`.
5. **Financial Metrics**: `Monthly Charge`, `Total Charges`, `Total Refunds`, `Total Extra Data Charges`, `Total Long Distance Charges`, `Total Revenue`.
6. **Post-Event Ground Truth (Leakage Sensitive)**: `Customer Status`, `Churn Category`, `Churn Reason`.

> [!CAUTION]
> **Data Leakage Hazard**: `Churn Category` and `Churn Reason` are collected *after* a customer departs. Including these columns in model training constitutes severe target leakage. They are strictly isolated and used solely for post-hoc validation of SHAP explanations.

---

## 🛠️ Layer-by-Layer Methodology

### Layer 1: Predictive Modeling
- **Data Preprocessing**:
  - Handle structural missing values in service-dependent fields (e.g., customers without Internet Service have null/NA for `Online Security`, `Streaming TV`, etc. These are imputed with informative categories like `"No Internet Service"` rather than blind imputation).
  - One-Hot and Target Encoding for high-cardinality nominals (`City`, `Zip Code`).
  - Standard scaling for continuous features (`Tenure`, `Monthly Charge`, `Total Revenue`).
- **Class Imbalance Remedies**:
  - SMOTE (Synthetic Minority Over-sampling Technique) on training folds.
  - Cost-sensitive classification using class weighting ($w_1 / w_0$).
  - Dynamic classification thresholds derived from utility matrices rather than arbitrary $0.5$ cutoffs.
- **Algorithms Evaluated**:
  1. **Logistic Regression** (L2-penalized baseline with interpretable odds ratios).
  2. **Random Forest Classifier** (Non-linear bagging ensemble).
  3. **XGBoost (Extreme Gradient Boosting)** (Optimized gradient-boosted decision trees).
- **Validation**:
  - Stratified 5-Fold Cross-Validation.
  - Metrics: ROC-AUC, PR-AUC, F1-Score, Recall (critical to avoid false negatives).

### Layer 2: Customer Lifetime Value (CLV) Modeling
Two complementary formulations are developed and benchmarked:
1. **Historical CLV ($CLV_{hist}$)**:
   $$\text{CLV}_{\text{hist}} = \text{Total Charges} - \text{Total Refunds} + \text{Total Extra Data} + \text{Total Long Distance}$$
2. **Predictive CLV ($CLV_{pred}$)**:
   $$\text{CLV}_{\text{pred}} = \text{Monthly Charge} \times \mathbb{E}[\text{Remaining Tenure}]$$
   Expected remaining tenure is modeled using a Kaplan-Meier survival curve or a parametric survival regression (Weibull / Cox Proportional Hazards) conditioned on customer tenure and contract type.

### Layer 3: Explainable AI (SHAP)
- Utilizes **TreeSHAP** (Lundberg & Lee, 2017) to calculate exact Shapley values with low computational complexity.
- **Global Interpretability**:
  - Beeswarm plots depicting feature impact directionality (e.g., does Fiber Optic increase or decrease churn probability?).
  - Feature importance ranking.
- **Local Interpretability**:
  - Individual waterfall charts for high-risk subscribers.
  - Automated extraction of top 3 churn drivers per flagged customer.

### Layer 4: Prescriptive Retention & Optimization
- **Value $\times$ Risk Prioritization Grid**:
  - **High Value, High Risk**: Immediate personal outreach, contract restructuring, rate discounts.
  - **High Value, Low Risk**: Loyalty rewards, proactive service checkups.
  - **Low Value, High Risk**: Automated email campaigns, self-serve retention offers.
  - **Low Value, Low Risk**: Maintain standard operating procedure.
- **Constrained Optimization Formulation**:
  Given a fixed retention budget $B$, select binary decision variables $x_i \in \{0, 1\}$ for each customer $i$:
  $$\max \sum_{i=1}^N x_i \cdot \Big[ P(\text{Churn}_i) \times \text{CLV}_i \times \Delta P_{\text{saved}} - C_{\text{intervention}} \Big]$$
  $$\text{Subject to: } \sum_{i=1}^N x_i \cdot C_{\text{intervention}} \le B$$
- **Algorithmic Fairness Audit**:
  - Statistical parity difference and equalized odds evaluated across demographic subgroups: `Age Band` ($<30, 30-50, 50+$), `Gender`, `Dependents`, `Married`.

---

## 📁 Repository Structure

```
AI_Project/
├── README.md                                  # Human-facing project documentation (this file)
├── AGENTS.md                                  # Autonomous AI agent operating manual & authoritative rules
├── PROGRESS.md                                # Real-time project milestones and phase roadmaps
├── F26-11.docx                                # Project Phase 1 proposal document
├── Dissertaion (AutoRecovered) ... (1).pdf    # Reference dissertation on telecom ML churn
├── Telecom+Customer+Churn/                    # Maven Analytics dataset files
│   ├── telecom_customer_churn.csv             # 7,043 rows × 38 columns raw dataset
│   ├── telecom_data_dictionary.csv            # Detailed feature definitions & types
│   └── telecom_zipcode_population.csv         # California zip code to population table
├── src/                                       # (Planned / In Development) Core Python source code
│   ├── data/                                  # Data ingestion, cleaning, feature pipelines
│   ├── models/                                # Model training, hyperparameter tuning, evaluation
│   ├── valuation/                             # Historical & predictive CLV calculators
│   ├── explainability/                        # SHAP calculation & plot generators
│   └── prescription/                          # Budget optimization solver & policy assigner
├── notebooks/                                 # Exploratory Data Analysis & experimentation
├── reports/                                   # LaTeX technical report, figures, and exports
└── dashboards/                                # Power BI / Streamlit retention dashboard assets
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12+ (tested on Python 3.14 macOS Apple Silicon / Linux)
- Recommended: Virtual environment (`venv` or `conda`)

### 1. Environment Setup
```bash
# Clone & enter repo
git clone https://github.com/HassanShakilPasha/telecom-value-based-retention.git
cd telecom-value-based-retention

# Setup environment
python -m venv venv
# Windows: .\venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt


# Install essential dependencies
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn scipy

#Run Pipeline
```

### 2. Verify Dataset Integrity
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('Telecom+Customer+Churn/telecom_customer_churn.csv')
print('Dataset shape:', df.shape)
print('Target counts:\n', df['Customer Status'].value_counts())
"
```

### 3. Run Phase 2 Data Preprocessing & Setup Pipeline
```bash
./venv/bin/python scripts/run_phase2_pipeline.py
```

### 4. Execute Automated Test Suite
```bash
./venv/bin/python -m unittest tests/test_data_pipeline.py
```

---

## 📈 Roadmap & Deliverables

> For a complete real-time tracking matrix and upcoming phase specifications, refer to [PROGRESS.md](file:///Users/hassanshakil/Antigravity/AI_Project/PROGRESS.md).

- [x] **Phase 1: Project Scoping & Problem Formulation**
  - [x] Gap analysis of binary churn vs. value-based retention
  - [x] Dataset curation & data dictionary mapping
  - [x] Team division & proposal documentation (`F26-11.docx`)
- [x] **Phase 2: Data Acquisition, Cleaning, Engineering & Setup**
  - [x] Merged California Zip Code population metrics
  - [x] Resolved structural service-dependency missing values
  - [x] Engineered add-on counts, financial ratios, and demographic bands
  - [x] Built hybrid leak-free pipeline (TargetEncoder for high-cardinality nominals, OneHotEncoder for low-cardinality, StandardScaler)
  - [x] Generated stratified partitions (`70% train`, `15% val`, `15% test`)
  - [x] Produced EDA figures and [Phase 2 Data Quality Audit Report](file:///Users/hassanshakil/Antigravity/AI_Project/reports/phase2_data_audit_report.md)
  - [x] Validated with automated test suite (`tests/test_data_pipeline.py`)
- [ ] **Phase 3: Predictive Modeling & Cost-Sensitive Learning**
  - [ ] Baseline Logistic Regression, Random Forest, XGBoost
  - [ ] 5-fold cross validation with ROC-AUC & PR-AUC tracking
  - [ ] CLV-weighted sample training & decision threshold tuning
- [ ] **Phase 4: Valuation & XAI Integration**
  - [ ] Historical vs. Predictive CLV modeling
  - [ ] Global & Local SHAP waterfall generation
  - [ ] Mapping churn drivers to retention offer taxonomy
- [ ] **Phase 5: Prescriptive Optimization & Fairness Audit**
  - [ ] Budget-constrained Knapsack / LP retention allocation
  - [ ] Disparate impact & fairness metrics on Age & Marital status
- [ ] **Phase 6: Final Deliverables**
  - [ ] Comprehensive LaTeX Technical Report
  - [ ] Power BI Executive Dashboard
  - [ ] Streamlit SHAP Retention Simulator (Bonus)

---

## 📚 Key Academic References

1. **Ahmad, A. K., Jafar, A., & Aljoumaa, K. (2019)**. *Customer churn prediction in telecom using machine learning in big data platform*. Journal of Big Data, 6(1), 1-24. [DOI: 10.1186/s40537-019-0191-6](https://doi.org/10.1186/s40537-019-0191-6)
2. **Jain, H., Khunteta, A., & Srivastava, S. (2020)**. *Churn prediction in telecommunication using Logistic Regression and Logit Boost*. Procedia Computer Science, 167, 101-112. [DOI: 10.1016/j.procs.2020.03.187](https://doi.org/10.1016/j.procs.2020.03.187)
3. **Coussement, K., & Van den Poel, D. (2008)**. *Churn prediction in subscription services: An application of support vector machines*. Expert Systems with Applications, 34(1), 313-327. [DOI: 10.1016/j.eswa.2006.09.038](https://doi.org/10.1016/j.eswa.2006.09.038)
4. **Lundberg, S. M., & Lee, S.-I. (2017)**. *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS 2017). [arXiv: 1705.07874](https://arxiv.org/abs/1705.07874)
5. **Chen, T., & Guestrin, C. (2016)**. *XGBoost: A Scalable Tree Boosting System*. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016). [DOI: 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

---

*Project developed under AI2002 Artificial Intelligence course guidelines. For questions or collaboration, please contact the team members.*
