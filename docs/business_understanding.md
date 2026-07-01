# Phase 1 — Business Understanding
## Problem Statement
TelCo Inc. lacks a proactive mechanism to identify at-risk customers before they churn, causing significant recurring revenue loss. This project builds an ML-powered churn prediction system paired with a Power BI dashboard to enable data-driven retention strategies.

## Business Objectives
1. Predict customers likely to churn in the next billing cycle
2. Quantify revenue at risk
3. Surface key churn drivers for the retention team
4. Enable priority outreach via a ranked risk list
5. Deliver insights through an executive BI dashboard

## Business Questions
- What is the overall churn rate?
- Which segments (contract type, payment method, tenure) churn most?
- How much monthly revenue is lost to churn?
- Which customers are highest risk RIGHT NOW?

## KPIs
| KPI | Definition | Target |
|-----|-----------|--------|
| Churn Rate | % customers lost in period | < 15% |
| Retention Rate | % customers retained | > 85% |
| Revenue Lost | Monthly revenue from churned customers | Minimize |
| CLV | Avg revenue × avg tenure | Maximize |
| High-Risk Count | Customers with churn probability > 70% | Track weekly |
| ROC-AUC | Model discrimination ability | ≥ 0.85 |

## Success Metrics
- ML model ROC-AUC ≥ 0.85
- Recall ≥ 0.80 (catching churners is priority)
- Dashboard adopted by retention & marketing teams
- ≥ 5 actionable retention recommendations delivered

## Business Value
| Stakeholder | Value |
|-------------|-------|
| CEO | Revenue protection, strategic roadmap |
| Marketing | Targeted retention campaigns |
| Customer Success | Priority outreach list |
| Finance | Revenue-at-risk quantification |

## Expected Financial Impact
> With 7,043 customers, avg charge $65/month, 26% churn rate:
> Monthly revenue at risk ≈ $118,000
> Reducing churn by 10% saves ≈ $141,600/year
