# EDA Summary Report
## AI-Powered Customer Churn Prediction
**Project:** IBM Telco Customer Churn  
**Phase:** 3 — Exploratory Data Analysis  
**Date:** 2025  
**Author:** [Your Name]

---

## Executive Summary

An EDA of 7,043 IBM Telco customers reveals a **22.1% churn rate** — above the telecom industry benchmark of 15–20%. Three factors dominate churn behaviour: **contract type**, **customer tenure**, and **payment method**. Month-to-month customers churn at 42%+ while 2-year contract customers churn at under 3%. The first 12 months of a customer relationship is the single highest-risk window. Estimated monthly revenue at risk from churn exceeds **$100,000**.

---

## Dataset Overview

| Property | Value |
|----------|-------|
| Rows | 7,043 |
| Columns | 23 (after cleaning) |
| Numerical Features | 3 (tenure, MonthlyCharges, TotalCharges) |
| Categorical Features | 16 |
| Missing Values | 0 (resolved in Phase 2) |
| Duplicate Rows | 0 |
| Memory Usage | 6.75 MB |
| Target: Churn Rate | 22.1% |
| Class Balance | 77.9% No-Churn / 22.1% Churn |

---

## Key Findings

### 1. Target Variable
- Churn rate: **22.1%** (1,555 churned out of 7,043)
- Class imbalance present → SMOTE required before ML training
- A naive "always predict No-Churn" model gets 77.9% accuracy but catches 0 churners

### 2. Numerical Features

| Feature | Mean | Median | Std Dev | Skewness |
|---------|------|--------|---------|----------|
| tenure | 35.9 mo | 36.0 mo | 20.95 | 0.00 |
| MonthlyCharges | $68.02 | $67.55 | $29.03 | 0.03 |
| TotalCharges | $2,309 | $1,883 | $1,773 | 0.89 |

**Churn vs Retained — Numerical Differences:**
- Churned avg tenure: ~18 months vs retained ~38 months
- Churned avg MonthlyCharges: ~$74 vs retained ~$61
- TotalCharges right-skewed — log-transform considered for linear models

### 3. Contract Type — Strongest Predictor
| Contract | Churn Rate |
|----------|-----------|
| Month-to-month | ~42% |
| One year | ~11% |
| Two year | ~3% |

**Finding:** Contract type alone can explain the majority of churn variance.

### 4. Payment Method
| Payment Method | Churn Rate |
|----------------|-----------|
| Electronic check | ~45% |
| Mailed check | ~19% |
| Bank transfer (auto) | ~17% |
| Credit card (auto) | ~15% |

**Finding:** Manual payment methods correlate strongly with churn. Auto-pay customers are significantly more loyal.

### 5. Internet Service
| Service Type | Churn Rate |
|-------------|-----------|
| Fiber optic | ~42% |
| DSL | ~19% |
| No internet | ~7% |

**Finding:** Fiber optic — the premium, highest-revenue product — has the worst retention. This is a critical business risk.

### 6. Tenure & Churn — Critical Window
- Customers in months 0–12: churn rate **~47%**
- Customers beyond 48 months: churn rate **< 6%**
- **Insight:** The first year is the make-or-break period for retention

### 7. Correlation Summary (Numerical vs Churn)
| Feature | Correlation with Churn | Interpretation |
|---------|----------------------|----------------|
| tenure | -0.35 | Longer tenure = much less likely to churn |
| TotalCharges | -0.20 | Higher lifetime value = more loyal |
| MonthlyCharges | +0.19 | Higher monthly cost = slightly higher churn risk |

### 8. Add-On Services Reduce Churn
Customers with Online Security, Tech Support, Device Protection, or Online Backup consistently churn less. Each add-on creates switching cost and perceived value.

### 9. Demographics
- **Senior Citizens:** churn at ~41% vs ~23% for non-seniors
- **Gender:** No meaningful difference (Male ≈ Female ≈ 22%)
- **No Partner:** churn at ~33% vs ~20% with partner
- **No Dependents:** churn at ~31% vs ~15% with dependents

### 10. Outlier Analysis
| Feature | Outliers | % | Business Decision |
|---------|----------|---|-------------------|
| tenure | 0 | 0% | N/A |
| MonthlyCharges | 0 | 0% | N/A |
| TotalCharges | ~267 | ~3.8% | **Keep** — high-value VIP customers |

---

## Business Recommendations

1. **Immediate:** Launch contract upgrade campaign for all month-to-month customers — offer 10% discount to switch to 1-year contracts
2. **Immediate:** Implement a 90-day onboarding programme with proactive check-ins at day 30, 60, and 90
3. **Short-term:** Incentivise auto-pay migration for electronic-check customers — every auto-pay conversion reduces churn risk by ~28 percentage points
4. **Short-term:** Audit fiber optic service quality — a 42% churn rate on the premium product is unacceptable
5. **Medium-term:** Develop a Senior Citizen support programme with dedicated helpline and simplified plan options
6. **Medium-term:** Bundle add-on services (Online Security, Tech Support) into base plans to increase stickiness
7. **Long-term:** Build a CLV-weighted retention investment model — not all customers deserve equal retention spend

---

## Limitations

- Dataset is cross-sectional (one point in time) — no time-series churn trajectory
- No customer satisfaction or NPS data available
- No competitor pricing context
- Churn reasons not captured (voluntary vs involuntary)
- Geographic data not available — regional patterns unknown

---

## Files Generated

### Figures (reports/figures/)
| File | Description |
|------|-------------|
| churn_distribution.png | Target variable pie + count chart |
| tenure_analysis.png | Histogram, KDE, boxplot, violin for tenure |
| monthlycharges_analysis.png | 4-chart numerical analysis |
| totalcharges_analysis.png | 4-chart numerical analysis |
| cat_contract_analysis.png | Contract stacked bar + churn rate |
| cat_paymentmethod_analysis.png | Payment method analysis |
| cat_internetservice_analysis.png | Internet service analysis |
| cat_seniorcitizenlabel_analysis.png | Senior citizen analysis |
| *(+16 more categorical charts)* | All 16 categorical features |
| contract_vs_churn.png | Focused bivariate: contract vs churn |
| payment_method_vs_churn.png | Payment method churn rates |
| internet_service_vs_churn.png | Internet type churn rates |
| seniorcitizen_vs_churn.png | Senior citizen churn analysis |
| gender_vs_churn.png | Gender churn comparison |
| partner_dependents_vs_churn.png | Household status analysis |
| tenure_vs_churn.png | Tenure histogram + KDE overlay |
| monthly_charges_vs_churn.png | Monthly charges distribution by churn |
| totalcharges_vs_churn.png | Total charges KDE by churn |
| correlation_heatmap.png | Full correlation matrix heatmap |

### Data Files
| File | Description |
|------|-------------|
| data/processed/telco_clean.csv | Cleaned dataset (Phase 2 output) |
| data/processed/powerbi_ready.csv | Power BI optimised export with extra columns |

---

## Next Steps (Phase 4 — Feature Engineering)

1. Encode all categorical variables (Label + OneHot)
2. Create `ServiceCount` feature (number of add-ons per customer)
3. Create `TenureGroup` bins (0–12, 13–24, 25–48, 49–72 months)
4. Create `ChargePerTenureMonth` ratio feature
5. Create `ContractRisk` numeric score
6. Scale numerical features for linear models
7. Apply SMOTE to address class imbalance

---
*Report generated by the EDA pipeline — `src/analysis/eda.py`*
