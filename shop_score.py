#!/usr/bin/env python3
"""
SHOP-SCORE: Advanced Customer Lifetime Value (CLV) & Churn Modeling
===================================================================
Domain : Data Analytics & Artificial Intelligence

End-to-end pipeline (single file):
  1. Load POS transactions (CSV) or generate a realistic synthetic supermarket dataset
  2. Behavioural scoring  -> RFM scores, segments (Champions ... Lost)
  3. Feature engineering  -> rolling spend, gap variability, basket variance, promo response
  4. Early-warning churn  -> Logistic Regression / Random Forest (/ XGBoost if installed),
                             recall-prioritised decision threshold
  5. 12-month CLV         -> Gradient Boosting regression (existing customers) and an
                             early-life model for newly acquired customers
  6. Deliverables         -> Executive dashboard (PNG), Intervention roster (CSV),
                             Acquisition budget report (CSV), metrics.json

Usage
-----
  python shop_score.py                              # synthetic demo data
  python shop_score.py --data transactions.csv      # your own POS data
  python shop_score.py --out results --customers 5000 --cac 350

Expected CSV columns: customer_id, date, amount   (optional: promo_used = 0/1)
"""
import argparse
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
CUR = "Rs"                       # currency label used in reports
DATA_START = pd.Timestamp("2023-01-01")
DATA_END = pd.Timestamp("2025-12-31")
TRAIN_CUTOFF = pd.Timestamp("2024-12-31")   # snapshot used to build training labels
CHURN_WINDOW = 90                # churn = no purchase in the next 90 days
CLV_HORIZON = 365                # 12-month CLV
TARGET_RECALL = 0.90             # churn model prioritises recall (few false negatives)
GROSS_MARGIN = 0.25              # supermarket gross margin used for ROI / promo limits
PROMO_CAP_PCT_OF_CLV = 0.10      # promo cost may never exceed 10% of 12m CLV
MAX_REDEMPTIONS = 3              # visits over which a discount code can be used
ACTIVE_MAX_RECENCY = 120         # customers silent longer than this are already "Lapsed"
EARLY_LIFE_DAYS = 60             # observation window for new-customer CLV model


# --------------------------------------------------------------------------- #
# 1. Data: load or simulate
# --------------------------------------------------------------------------- #
def simulate_transactions(n_customers=4000, seed=SEED):
    """Synthetic POS log with realistic behaviour: heterogeneous visit rates, basket
    sizes, promo response, random churn and a gradual slowdown before churn."""
    rng = np.random.default_rng(seed)
    span = (DATA_END - DATA_START).days
    rows = []
    for cid in range(1, n_customers + 1):
        acq = DATA_START + pd.Timedelta(days=int(rng.integers(0, span - 20)))
        lam = rng.gamma(2.0, 1.6) / 30.0 + 0.01          # visits per day
        basket = rng.lognormal(mean=6.1, sigma=0.45)      # typical basket (Rs)
        promo_p = rng.beta(2, 5)
        churn_h = rng.choice([0.0015, 0.004, 0.010], p=[0.35, 0.40, 0.25])
        churn_day = rng.exponential(1 / churn_h)
        horizon = min((DATA_END - acq).days, churn_day)
        n_draw = int(lam * (horizon + 1) * 2) + 5
        t = np.cumsum(rng.exponential(1 / lam, n_draw))
        t = t[t <= horizon]
        if len(t) == 0:
            t = np.array([0.0])
        t = np.concatenate([[0.0], t[t > 0]])             # first visit = acquisition
        if churn_day < (DATA_END - acq).days:             # pre-churn slowdown
            frac = np.clip((t - (churn_day - 75)) / 75, 0, 1)
            keep = rng.random(len(t)) > 0.65 * frac
            keep[0] = True
            t, frac = t[keep], frac[keep]
        else:
            frac = np.zeros(len(t))
        amt = basket * rng.lognormal(0, 0.35, len(t)) * (1 - 0.30 * frac)
        promo = (rng.random(len(t)) < promo_p).astype(int)
        amt = amt * (1 + 0.06 * promo)
        dates = acq + pd.to_timedelta(np.floor(t), unit="D")
        rows.append(pd.DataFrame({"customer_id": cid, "date": dates,
                                  "amount": np.round(amt, 2), "promo_used": promo}))
    tx = pd.concat(rows, ignore_index=True)
    return tx[tx.date <= DATA_END].sort_values(["customer_id", "date"]).reset_index(drop=True)


def load_transactions(path=None, n_customers=4000):
    if path:
        tx = pd.read_csv(path, parse_dates=["date"])
        need = {"customer_id", "date", "amount"}
        if not need.issubset(tx.columns):
            raise ValueError(f"CSV must contain columns {need}")
    else:
        tx = simulate_transactions(n_customers)
    if "promo_used" not in tx.columns:
        tx["promo_used"] = 0
    tx = tx.dropna(subset=["customer_id", "date", "amount"])
    tx = tx[tx.amount > 0].drop_duplicates()
    tx["date"] = pd.to_datetime(tx["date"]).dt.normalize()
    return tx.sort_values(["customer_id", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Feature engineering + RFM
# --------------------------------------------------------------------------- #
FEATURES = ["recency", "frequency", "monetary", "tenure", "avg_basket", "basket_std",
            "basket_cv", "mean_gap", "std_gap", "recency_ratio", "f90", "f_prev90",
            "freq_trend", "spend_90", "spend_prev90", "spend_trend", "basket_ratio",
            "roll_spend_3m", "roll_spend_6m", "promo_rate"]


def build_features(tx, asof):
    """Customer-level feature table using only information available at `asof`."""
    h = tx[tx.date <= asof].copy()
    h["gap"] = h.groupby("customer_id").date.diff().dt.days
    g = h.groupby("customer_id")
    f = pd.DataFrame({
        "recency": (asof - g.date.max()).dt.days,
        "frequency": g.size(),
        "monetary": g.amount.sum(),
        "tenure": (asof - g.date.min()).dt.days,
        "basket_std": g.amount.std().fillna(0),
        "mean_gap": g.gap.mean(),
        "std_gap": g.gap.std(),
        "promo_rate": g.promo_used.mean(),
    })
    f["avg_basket"] = f.monetary / f.frequency
    f["basket_cv"] = f.basket_std / f.avg_basket
    f["mean_gap"] = f.mean_gap.fillna(f.tenure.clip(lower=30))
    f["std_gap"] = f.std_gap.fillna(0)
    f["recency_ratio"] = f.recency / (f.mean_gap + 1)

    def window(lo, hi):
        w = h[(h.date > asof - pd.Timedelta(days=hi)) & (h.date <= asof - pd.Timedelta(days=lo))]
        gw = w.groupby("customer_id")
        return gw.size(), gw.amount.sum(), gw.amount.mean()

    (n90, s90, b90), (np90, sp90, _) = window(0, 90), window(90, 180)
    _, s180, _ = window(0, 180)
    _, s90b, _ = window(0, 90)
    f["f90"] = n90.reindex(f.index).fillna(0)
    f["f_prev90"] = np90.reindex(f.index).fillna(0)
    f["spend_90"] = s90.reindex(f.index).fillna(0)
    f["spend_prev90"] = sp90.reindex(f.index).fillna(0)
    f["freq_trend"] = (f.f90 + 1) / (f.f_prev90 + 1)
    f["spend_trend"] = (f.spend_90 + 1) / (f.spend_prev90 + 1)
    f["basket_ratio"] = (b90.reindex(f.index).fillna(f.avg_basket)) / f.avg_basket
    f["roll_spend_3m"] = f.spend_90 / 3
    f["roll_spend_6m"] = s180.reindex(f.index).fillna(0) / 6
    return f


def rfm_segments(f):
    """Quintile RFM scores (5 = best), segment names and a rule-based risk score."""
    s = pd.DataFrame(index=f.index)
    s["R"] = pd.qcut((-f.recency).rank(method="first"), 5, labels=False) + 1
    s["F"] = pd.qcut(f.frequency.rank(method="first"), 5, labels=False) + 1
    s["M"] = pd.qcut(f.monetary.rank(method="first"), 5, labels=False) + 1

    def name(r):
        if r.R >= 4 and r.F >= 4:
            return "Champions"
        if r.R >= 3 and r.F >= 3:
            return "Loyal"
        if r.R >= 4 and r.F <= 2:
            return "New / Promising"
        if r.R <= 2 and r.F >= 3:
            return "At-Risk"
        if r.R == 1 and r.F <= 2:
            return "Lost"
        return "Needs Attention"

    s["segment"] = s.apply(name, axis=1)
    s["rfm_score"] = s.R + s.F + s.M
    s["rfm_risk"] = np.round(100 * (1 - (0.5 * s.R + 0.3 * s.F + 0.2 * s.M - 1) / 4), 1)
    return s


def make_labels(tx, f, asof):
    fut = tx[(tx.date > asof) & (tx.date <= asof + pd.Timedelta(days=CHURN_WINDOW))]
    active = set(fut.customer_id)
    churn = (~f.index.isin(active)).astype(int)
    fut12 = tx[(tx.date > asof) & (tx.date <= asof + pd.Timedelta(days=CLV_HORIZON))]
    clv = fut12.groupby("customer_id").amount.sum().reindex(f.index).fillna(0)
    return pd.Series(churn, index=f.index, name="churn"), clv.rename("clv_12m")


# --------------------------------------------------------------------------- #
# 3. Churn model
# --------------------------------------------------------------------------- #
def train_churn(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    models = {
        "LogisticRegression": make_pipeline(StandardScaler(),
                                            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                               class_weight="balanced", n_jobs=-1, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        pos = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, scale_pos_weight=pos,
                                          eval_metric="logloss", random_state=SEED)
    except Exception:
        pass

    results, fitted = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        results[name] = {"roc_auc": roc_auc_score(yte, p),
                         "pr_auc": average_precision_score(yte, p)}
        fitted[name] = (m, p)
    best = max(results, key=lambda k: results[k]["pr_auc"])
    model, p_test = fitted[best]

    # recall-prioritised threshold chosen on out-of-fold predictions (no test leakage)
    oof = cross_val_predict(models[best].__class__(**models[best].get_params())
                            if best != "LogisticRegression" else models[best],
                            Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(ytr, oof)
    ok = np.where(rec[:-1] >= TARGET_RECALL)[0]
    threshold = float(thr[ok[-1]]) if len(ok) else 0.5

    pred = (p_test >= threshold).astype(int)
    metrics = {
        "best_model": best, "threshold": round(threshold, 3),
        "precision": precision_score(yte, pred), "recall": recall_score(yte, pred),
        "f1": f1_score(yte, pred), "roc_auc": results[best]["roc_auc"],
        "pr_auc": results[best]["pr_auc"], "churn_rate": float(y.mean()),
        "model_comparison": results,
    }
    curve = precision_recall_curve(yte, p_test)
    return model, threshold, metrics, curve


# --------------------------------------------------------------------------- #
# 4. CLV models
# --------------------------------------------------------------------------- #
def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def train_clv(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=SEED)
    model.fit(Xtr, ytr)
    p = np.clip(model.predict(Xte), 0, None)
    base = Xte.roll_spend_6m * 12          # naive baseline: annualised trailing spend
    m = {"rmse": rmse(yte, p), "mae": float(mean_absolute_error(yte, p)),
         "baseline_rmse": rmse(yte, base), "baseline_mae": float(mean_absolute_error(yte, base)),
         "mean_actual_clv": float(yte.mean())}
    return model, m


def early_life_features(tx, first_dates, days=EARLY_LIFE_DAYS):
    d = tx.merge(first_dates.rename("first"), left_on="customer_id", right_index=True)
    d["age"] = (d.date - d["first"]).dt.days
    e = d[d.age <= days]
    g = e.groupby("customer_id")
    x = pd.DataFrame({"visits": g.size(), "spend": g.amount.sum(),
                      "avg_basket": g.amount.mean(), "promo_rate": g.promo_used.mean()})
    second = e[e.age > 0].groupby("customer_id").age.min()
    x["days_to_2nd"] = second.reindex(x.index).fillna(days + 30)
    return x


def acquisition_report(tx, cac, out):
    """Early-life CLV model + cohort-level ROI vs. acquisition spend."""
    first = tx.groupby("customer_id").date.min()
    hist = first[first <= DATA_END - pd.Timedelta(days=CLV_HORIZON)].index
    X = early_life_features(tx, first).reindex(hist).dropna()
    d = tx.merge(first.rename("first"), left_on="customer_id", right_index=True)
    d["age"] = (d.date - d["first"]).dt.days
    y = d[d.age <= CLV_HORIZON].groupby("customer_id").amount.sum().reindex(X.index).fillna(0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    mdl = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=SEED).fit(Xtr, ytr)
    p = np.clip(mdl.predict(Xte), 0, None)
    metrics = {"rmse": rmse(yte, p), "mae": float(mean_absolute_error(yte, p)),
               "baseline_rmse": rmse(yte, np.full(len(yte), ytr.mean())),
               "baseline_mae": float(mean_absolute_error(yte, np.full(len(yte), ytr.mean())))}

    # cohorts acquired in 2025 with >= 60 days of observed behaviour
    new = first[(first >= pd.Timestamp("2025-01-01")) &
                (first <= DATA_END - pd.Timedelta(days=EARLY_LIFE_DAYS))].index
    Xn = early_life_features(tx, first).reindex(new).dropna()
    pred = pd.Series(np.clip(mdl.predict(Xn), 0, None), index=Xn.index)
    coh = first.reindex(Xn.index).dt.to_period("Q").astype(str)
    rep = pd.DataFrame({"cohort": coh, "pred_rev": pred}).groupby("cohort").agg(
        new_customers=("pred_rev", "size"), avg_pred_12m_revenue=("pred_rev", "mean"),
        total_pred_12m_revenue=("pred_rev", "sum")).reset_index()
    rep["acquisition_spend"] = rep.new_customers * cac
    rep["predicted_gross_profit"] = rep.total_pred_12m_revenue * GROSS_MARGIN
    rep["roi_pct"] = 100 * (rep.predicted_gross_profit - rep.acquisition_spend) / rep.acquisition_spend
    rep["revenue_to_cac"] = rep.total_pred_12m_revenue / rep.acquisition_spend
    rep["max_affordable_cac"] = rep.avg_pred_12m_revenue * GROSS_MARGIN   # break-even cap
    rep["roi_positive"] = rep.roi_pct > 0
    rep = rep.round(2)
    rep.to_csv(os.path.join(out, "acquisition_budget_report.csv"), index=False)
    return rep, metrics


# --------------------------------------------------------------------------- #
# 5. Deliverables
# --------------------------------------------------------------------------- #
def intervention_roster(scored, threshold, out):
    r = scored[(scored.churn_prob >= threshold) & (scored.recency <= ACTIVE_MAX_RECENCY)
               & (scored.clv_12m > 0)].copy()
    q = r.clv_12m.rank(pct=True)
    r["offer_tier_pct"] = np.select([q > 0.66, q > 0.33], [0.15, 0.10], 0.05)
    offer = r.offer_tier_pct * r.avg_basket * MAX_REDEMPTIONS
    r["max_promo_cost"] = np.round(np.minimum(offer, PROMO_CAP_PCT_OF_CLV * r.clv_12m), 2)
    r["discount_pct_per_visit"] = np.round(
        100 * r.max_promo_cost / (r.avg_basket * MAX_REDEMPTIONS), 1)
    r["channel"] = np.where(r.promo_rate > 0.25, "SMS + app coupon", "Email")
    r["expected_value_at_risk"] = np.round(r.churn_prob * r.clv_12m, 2)
    assert (r.max_promo_cost <= r.clv_12m).all()          # cost never exceeds CLV
    cols = ["segment", "churn_prob", "recency", "clv_12m", "avg_basket",
            "discount_pct_per_visit", "max_promo_cost", "channel", "expected_value_at_risk"]
    r = r.sort_values("expected_value_at_risk", ascending=False)[cols].round(3)
    r.index.name = "customer_id"
    r.to_csv(os.path.join(out, "intervention_roster.csv"))
    return r


def dashboard(scored, thr, churn_m, clv_m, trend, acq, curve, out):
    fig = plt.figure(figsize=(16, 10.5))
    fig.suptitle("SHOP-SCORE  |  Executive Retention Dashboard", fontsize=18, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28, top=0.9, bottom=0.07)

    # KPIs
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    A = scored[scored.risk_tier != "Lapsed"]
    hv = A.clv_12m >= A.clv_12m.quantile(0.75)
    hv_risk = 100 * (hv & (A.risk_tier == "High")).sum() / max(hv.sum(), 1)
    kpis = [("Active customers (last visit <=120d)", f"{len(A):,}"),
            ("Active customers in High-risk tier", f"{100 * (A.risk_tier == 'High').mean():.1f}%"),
            ("High-value customers at High risk", f"{hv_risk:.1f}%"),
            ("Avg predicted 12m CLV (active)", f"{CUR} {A.clv_12m.mean():,.0f}"),
            ("Revenue at risk (p x CLV)", f"{CUR} {(A.churn_prob * A.clv_12m).sum():,.0f}"),
            ("Churn model recall / precision", f"{churn_m['recall']:.2f} / {churn_m['precision']:.2f}")]
    for i, (k, v) in enumerate(kpis):
        y = 0.93 - i * 0.16
        ax.text(0, y, k, fontsize=11, color="#555")
        ax.text(0, y - 0.07, v, fontsize=17, fontweight="bold", color="#1f3b73")
    ax.set_title("Key metrics", loc="left", fontweight="bold")

    # Segments
    ax = fig.add_subplot(gs[0, 1])
    seg = scored.segment.value_counts()
    ax.barh(seg.index[::-1], seg.values[::-1], color="#3b6fb6")
    ax.set_title("RFM segments", fontweight="bold"); ax.set_xlabel("Customers")

    # Risk tiers
    ax = fig.add_subplot(gs[0, 2])
    order = ["Low", "Medium", "High", "Lapsed"]
    t = scored.risk_tier.value_counts().reindex(order).fillna(0)
    ax.bar(order, t.values, color=["#4caf50", "#ffb300", "#e53935", "#9e9e9e"])
    for i, v in enumerate(t.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom")
    ax.set_title("Retention risk tiers", fontweight="bold"); ax.set_ylabel("Customers")

    # CLV trend
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(trend.index, trend.values, marker="o", color="#1f3b73")
    ax.set_title("Avg predicted 12m CLV over time", fontweight="bold")
    ax.set_ylabel(f"{CUR} per active customer"); ax.tick_params(axis="x", rotation=30)

    # PR curve
    ax = fig.add_subplot(gs[1, 1])
    prec, rec, _ = curve
    ax.plot(rec, prec, color="#c62828")
    ax.axvline(TARGET_RECALL, ls="--", color="grey")
    ax.set_title(f"Churn model PR curve ({churn_m['best_model']})", fontweight="bold")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")

    # Acquisition
    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(len(acq)); w = 0.38
    ax.bar(x - w / 2, acq.acquisition_spend / 1e3, w, label="Acquisition spend", color="#9e9e9e")
    ax.bar(x + w / 2, acq.predicted_gross_profit / 1e3, w, label="Pred. 12m gross profit",
           color="#2e7d32")
    ax.set_xticks(x); ax.set_xticklabels(acq.cohort)
    ax.set_title("New-customer cohorts: spend vs profit", fontweight="bold")
    ax.set_ylabel(f"{CUR} '000"); ax.set_ylim(0, ax.get_ylim()[1] * 1.2); ax.legend(fontsize=8, loc="upper right")
    fig.savefig(os.path.join(out, "executive_dashboard.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="SHOP-SCORE CLV & churn pipeline")
    ap.add_argument("--data", help="CSV of POS transactions (customer_id,date,amount[,promo_used])")
    ap.add_argument("--out", default="shop_score_outputs")
    ap.add_argument("--customers", type=int, default=4000, help="synthetic customers if no --data")
    ap.add_argument("--cac", type=float, default=1200.0, help=f"acquisition cost per customer ({CUR})")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("[1/6] Loading data ...")
    tx = load_transactions(a.data, a.customers)
    print(f"      {len(tx):,} transactions, {tx.customer_id.nunique():,} customers")

    print("[2/6] Training churn model ...")
    f_train = build_features(tx, TRAIN_CUTOFF)
    f_train = f_train[f_train.recency <= ACTIVE_MAX_RECENCY]   # model only still-active customers
    y_churn, y_clv = make_labels(tx, f_train, TRAIN_CUTOFF)
    churn_model, thr, churn_m, curve = train_churn(f_train[FEATURES], y_churn)
    print(f"      best={churn_m['best_model']}  recall={churn_m['recall']:.3f} "
          f"precision={churn_m['precision']:.3f} F1={churn_m['f1']:.3f}")

    print("[3/6] Training 12-month CLV model ...")
    clv_model, clv_m = train_clv(f_train[FEATURES], y_clv)
    print(f"      RMSE={clv_m['rmse']:.0f} MAE={clv_m['mae']:.0f} "
          f"(baseline RMSE={clv_m['baseline_rmse']:.0f})")

    print("[4/6] Scoring customers as of", DATA_END.date(), "...")
    f_now = build_features(tx, DATA_END)
    scored = f_now.join(rfm_segments(f_now))
    act = scored.recency <= ACTIVE_MAX_RECENCY
    scored["churn_prob"] = 1.0
    scored["clv_12m"] = 0.0
    scored.loc[act, "churn_prob"] = churn_model.predict_proba(f_now.loc[act, FEATURES])[:, 1]
    scored.loc[act, "clv_12m"] = np.clip(clv_model.predict(f_now.loc[act, FEATURES]), 0, None)
    scored["risk_tier"] = np.select(
        [~act, scored.churn_prob >= thr, scored.churn_prob >= thr / 2],
        ["Lapsed", "High", "Medium"], "Low")
    scored.index.name = "customer_id"
    scored.round(3).to_csv(os.path.join(a.out, "customer_scores.csv"))

    print("[5/6] Building deliverables ...")
    roster = intervention_roster(scored, thr, a.out)
    acq, acq_m = acquisition_report(tx, a.cac, a.out)
    trend = {}
    for q in pd.date_range("2024-06-30", DATA_END, freq="QE"):
        ff = build_features(tx, q)
        ff = ff[ff.recency <= ACTIVE_MAX_RECENCY]
        trend[q.strftime("%Y-%m")] = np.clip(clv_model.predict(ff[FEATURES]), 0, None).mean()
    trend = pd.Series(trend)
    dashboard(scored, thr, churn_m, clv_m, trend, acq, curve, a.out)

    print("[6/6] Saving metrics ...")
    A = scored[scored.risk_tier != "Lapsed"]
    hv = A.clv_12m >= A.clv_12m.quantile(0.75)
    summary = {
        "customers": int(len(scored)), "transactions": int(len(tx)),
        "churn_model": churn_m, "clv_model": clv_m, "early_life_clv_model": acq_m,
        "active_customers": int(len(A)), "lapsed_customers": int(len(scored) - len(A)),
        "high_risk_pct": round(100 * float((A.risk_tier == "High").mean()), 2),
        "high_value_in_high_risk_pct": round(100 * float((hv & (A.risk_tier == "High")).sum() / hv.sum()), 2),
        "avg_clv_12m": round(float(A.clv_12m.mean()), 2),
        "revenue_at_risk": round(float((A.churn_prob * A.clv_12m).sum()), 2),
        "roster_size": int(len(roster)), "roster_total_promo_cap": round(float(roster.max_promo_cost.sum()), 2),
        "roster_value_at_risk": round(float(roster.expected_value_at_risk.sum()), 2),
        "segments": scored.segment.value_counts().to_dict(),
        "clv_trend": {k: round(v, 2) for k, v in trend.items()},
        "acquisition": acq.to_dict(orient="records"), "cac_assumed": a.cac,
    }
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"Done. Outputs in ./{a.out}/")


if __name__ == "__main__":
    main()
