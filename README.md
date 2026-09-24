# SHOP-SCORE: Advanced Customer Lifetime Value (CLV) & Churn Modeling

**Domain:** Data Analytics & Artificial Intelligence

SHOP-SCORE moves supermarket analytics from demographic segmentation to **predictive, AI-driven retention**.
From POS transaction logs it (1) scores customers with RFM, (2) predicts churn early, and (3) forecasts
12-month CLV, then turns those predictions into three business deliverables.

## Features
| Module | What it does |
|---|---|
| Behavioural scoring (RFM) | Recency / Frequency / Monetary quintile scores, segments (Champions, Loyal, At-Risk, Lost, ...) and a rule-based risk score |
| Early-warning churn model | Logistic Regression vs Random Forest (XGBoost if installed). Churn = no purchase in next 90 days. Threshold tuned for **recall >= 0.90** on out-of-fold data |
| 12-month CLV | Gradient Boosting regression for existing customers + an *early-life* model (first 60 days) for newly acquired customers |

Engineered features: rolling 3/6-month spend, std-dev of days between purchases, basket-size variance,
frequency/spend trend (last 90d vs previous 90d), recency ratio, promo response rate.

## Deliverables (written to `shop_score_outputs/`)
- `executive_dashboard.png` - Executive Retention Dashboard
- `intervention_roster.csv` - churn alerts with recommended discount, channel and a promo cost cap (always <= 10% of CLV)
- `acquisition_budget_report.csv` - predicted 12-month revenue/profit vs acquisition spend per cohort
- `customer_scores.csv` - RFM, churn probability, CLV and risk tier for every customer
- `metrics.json` - all evaluation metrics

## Quick start
```bash
git clone <your-repo-url>
cd shop-score
pip install -r requirements.txt

python shop_score.py                            # runs on built-in synthetic supermarket data
python shop_score.py --data my_pos_data.csv     # runs on your data
python shop_score.py --cac 1500 --out results   # custom acquisition cost per customer
```

### Input format
CSV with columns `customer_id, date, amount` and optionally `promo_used` (0/1).

## Evaluation (synthetic demo data, 4,000 customers, ~100k transactions)
| Model | Metric | Result |
|---|---|---|
| Churn (Random Forest) | Recall / Precision / F1 | 0.89 / 0.58 / 0.70 |
| Churn | ROC-AUC | 0.91 |
| 12m CLV (Gradient Boosting) | RMSE / MAE | 11,268 / 7,456 (baseline 16,213 / 10,021) |

Recall is prioritised on purpose: missing a customer who is about to leave (false negative) costs more than
sending an unnecessary discount (false positive), and the promo cap bounds the cost of the latter.

## Project structure
```
shop-score/
├── shop_score.py        # full pipeline (single file)
├── requirements.txt
├── README.md
├── SHOP-SCORE_Project_Report.docx
└── sample_outputs/      # dashboard, roster, acquisition report, metrics
```

## Configuration
Top of `shop_score.py`: churn window, recall target, gross margin, promo cap, active-customer window.

## Notes
- The demo data is simulated; metrics on real POS data will differ.
- Point-in-time features are used for all training snapshots, so no future information leaks into the models.
