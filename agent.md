# AI Agent Operating Manual & Project Context

> **Note**: This file is an exact operational guide for AI agents working in this repository. The primary standard specification is also documented in [AGENTS.md](file:///Users/hassanshakil/Antigravity/AI_Project/AGENTS.md) and [README.md](file:///Users/hassanshakil/Antigravity/AI_Project/README.md).

---

## 1. Project Mission & Identity

- **Course**: AI2002 Artificial Intelligence
- **Track**: Track A: Research & Development (R&D)
- **Group ID**: `F26-11`
- **Team**:
  - Saqib Qazi (`24L-2506`)
  - Hassan Shakil (`24L-2521`)
  - Ammara Saleem (`24L-2571`)
- **Project Title**: *From Prediction to Prescription: An Explainable AI Framework for Value-Based Customer Retention in the Telecom Sector*

### Research Objective
Bridge the gap between pure statistical churn prediction and actionable business decision-making through a 4-layer framework:
1. **Prediction Layer**: ML models (Logistic Regression, Random Forest, XGBoost) with cost-sensitive thresholding.
2. **Valuation Layer**: Historical and Predictive Customer Lifetime Value (CLV).
3. **Explainability Layer**: SHAP global and local feature importance.
4. **Prescriptive Action Layer**: Budget-constrained optimization and algorithmic fairness audit.

---

## 2. Critical Agent Guardrails & Invariants

1. **TARGET LEAKAGE PROHIBITION**:
   - `Churn Category` and `Churn Reason` MUST NEVER be used as features during model training or validation. They are post-churn labels.
2. **TARGET VARIABLE DEFINITION**:
   - Column: `Customer Status`
   - `Churned` = 1 (Positive / Churn)
   - `Stayed` = 0 (Negative / Retained)
   - `Joined` = 0 (Negative / Retained — newly joined Q2 2022 customers)
3. **SERVICE-DEPENDENCY NULL HANDLING**:
   - Features like `Online Security`, `Streaming TV`, etc. are null when `Internet Service == 'No'`.
   - Do NOT drop rows or blindly impute with mode/median.
   - Impute with `"No Internet Service"` or explicit indicator categories.
4. **PREPROCESSING SPLIT INTEGRITY**:
   - Perform stratified splits (`random_state=42`).
   - Fit scalers and categorical encoders ONLY on the training split to prevent test set data leakage.

---

## 3. Mathematical Reference

### Cost-Sensitive Decision Threshold
$$\tau_i^* = \frac{C_{\text{offer}}}{P(\text{retention succeeds}) \times \text{CLV}_i}$$

### Historical CLV Formula
$$\text{CLV}_{\text{hist}} = \text{Total Charges} - \text{Total Refunds} + \text{Total Extra Data Charges} + \text{Total Long Distance Charges} \equiv \text{Total Revenue}$$

### Predictive CLV Formula
$$\text{CLV}_{\text{pred}} = \text{Monthly Charge} \times \hat{T}_{\text{remaining}}$$

### Prescriptive Optimization (0-1 Knapsack Formulation)
$$\max_{\{x_i \in \{0, 1\}\}} \sum_{i=1}^N x_i \cdot \Big[ P(\text{Churn}_i) \times \text{CLV}_i \times \delta - c_i \Big] \quad \text{s.t.} \quad \sum_{i=1}^N x_i \cdot c_i \le B$$
Where:
- $x_i$: Retention intervention flag for customer $i$
- $\delta$: Retention campaign success rate (e.g., 0.35)
- $c_i$: Cost of intervention
- $B$: Total available marketing/retention budget

---

## 4. Key Files in Workspace
- [README.md](file:///Users/hassanshakil/Antigravity/AI_Project/README.md): Human-oriented project documentation & roadmap.
- [AGENTS.md](file:///Users/hassanshakil/Antigravity/AI_Project/AGENTS.md): Full technical rules, schema table, and agent specifications.
- [F26-11.docx](file:///Users/hassanshakil/Antigravity/AI_Project/F26-11.docx): Official Phase 1 submission & problem proposal.
- [telecom_customer_churn.csv](file:///Users/hassanshakil/Antigravity/AI_Project/Telecom+Customer+Churn/telecom_customer_churn.csv): Primary dataset (7,043 customers × 38 attributes).
- [telecom_data_dictionary.csv](file:///Users/hassanshakil/Antigravity/AI_Project/Telecom+Customer+Churn/telecom_data_dictionary.csv): Column definitions & descriptions.
- [telecom_zipcode_population.csv](file:///Users/hassanshakil/Antigravity/AI_Project/Telecom+Customer+Churn/telecom_zipcode_population.csv): California zipcode population lookup.
