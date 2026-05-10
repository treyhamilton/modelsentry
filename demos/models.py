"""Three realistic simulated models for the ModelSentry demo.

Each model exposes a `generate(state, n)` function that returns
(features_df, predictions_array). State is "baseline", "warning", or "critical"
— matching the dashboard severity tiers driven by PSI thresholds (0.10 / 0.25).
Warning-tier shifts are tuned so dynamic features land at PSI ~0.10–0.20 and
critical-tier shifts at PSI well past 0.25.

Realistic for a 50-200 person SaaS company: churn prediction (retention),
lead scoring (sales), fraud detection (payments).

These are NOT real ML models — predictions are simple rules-of-thumb so the
demo focuses on feature drift detection, which is what ModelSentry actually
catches.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

State = Literal["baseline", "warning", "critical"]

_RNG = np.random.default_rng()


# ---------------------------------------------------------------------------
# churn-v3: binary classification (cancel / retain)
# ---------------------------------------------------------------------------

_CHURN_TIERS_BASELINE = ["starter", "pro", "growth", "enterprise"]
_CHURN_TIER_P_BASELINE = [0.30, 0.50, 0.18, 0.02]
_CHURN_TIER_P_WARNING  = [0.45, 0.40, 0.13, 0.02]
_CHURN_TIER_P_CRITICAL = [0.70, 0.22, 0.07, 0.01]

_CHURN_COUNTRIES = ["US", "UK", "CA", "DE", "FR"]
_CHURN_COUNTRY_P = [0.60, 0.15, 0.10, 0.08, 0.07]


def generate_churn(state: State, n: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    if state == "baseline":
        tenure = _RNG.normal(24, 14, n).clip(0, 120)
        revenue = _RNG.normal(180, 95, n).clip(0, 2000)
        tier_p = _CHURN_TIER_P_BASELINE
    elif state == "warning":
        tenure = _RNG.normal(20, 14, n).clip(0, 120)
        revenue = _RNG.normal(150, 90, n).clip(0, 2000)
        tier_p = _CHURN_TIER_P_WARNING
    else:  # critical
        tenure = _RNG.normal(9, 8, n).clip(0, 120)
        revenue = _RNG.normal(65, 40, n).clip(0, 2000)
        tier_p = _CHURN_TIER_P_CRITICAL

    tickets = _RNG.poisson(1.3, n).astype(float)
    plan_tier = _RNG.choice(_CHURN_TIERS_BASELINE, size=n, p=tier_p)
    country = _RNG.choice(_CHURN_COUNTRIES, size=n, p=_CHURN_COUNTRY_P)

    df = pd.DataFrame({
        "tenure_months": tenure.round(1),
        "monthly_revenue": revenue.round(2),
        "support_tickets_30d": tickets,
        "plan_tier": plan_tier,
        "country": country,
    })

    # Toy churn rule: shorter tenure + low revenue + high tickets → cancel
    risk = (24 - df["tenure_months"]) / 24 + (200 - df["monthly_revenue"]) / 200 + df["support_tickets_30d"] / 5
    preds = (risk > 1.0).astype(int).to_numpy()
    return df, preds


# ---------------------------------------------------------------------------
# lead-score-v2: regression (score 0-100)
# ---------------------------------------------------------------------------

_LEAD_INDUSTRIES = ["tech", "finance", "healthcare", "retail", "other"]
_LEAD_INDUSTRY_P = [0.40, 0.25, 0.15, 0.12, 0.08]

_LEAD_SOURCES = ["organic", "paid", "referral", "event"]
_LEAD_SOURCE_P_BASELINE = [0.40, 0.25, 0.20, 0.15]
_LEAD_SOURCE_P_WARNING  = [0.27, 0.45, 0.16, 0.12]
_LEAD_SOURCE_P_CRITICAL = [0.15, 0.65, 0.12, 0.08]


def generate_lead(state: State, n: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    if state == "baseline":
        company_size = _RNG.lognormal(np.log(50), 0.9, n).clip(1, 5000)
        pageviews = _RNG.normal(8.5, 6, n).clip(0, 100)
        trial_days = _RNG.normal(4.5, 3, n).clip(0, 14)
        source_p = _LEAD_SOURCE_P_BASELINE
    elif state == "warning":
        company_size = _RNG.lognormal(np.log(35), 0.9, n).clip(1, 5000)
        pageviews = _RNG.normal(7.0, 5.5, n).clip(0, 100)
        trial_days = _RNG.normal(3.8, 2.8, n).clip(0, 14)
        source_p = _LEAD_SOURCE_P_WARNING
    else:  # critical
        company_size = _RNG.lognormal(np.log(20), 0.9, n).clip(1, 5000)
        pageviews = _RNG.normal(3.2, 2.5, n).clip(0, 100)
        trial_days = _RNG.normal(1.8, 2, n).clip(0, 14)
        source_p = _LEAD_SOURCE_P_CRITICAL

    industry = _RNG.choice(_LEAD_INDUSTRIES, size=n, p=_LEAD_INDUSTRY_P)
    source = _RNG.choice(_LEAD_SOURCES, size=n, p=source_p)

    df = pd.DataFrame({
        "company_size": company_size.round(0),
        "industry": industry,
        "source": source,
        "num_pageviews": pageviews.round(1),
        "trial_days_used": trial_days.round(1),
    })

    # Toy lead score: bigger companies + more engagement = higher score
    score = (
        20 * np.log1p(df["company_size"]) / 5
        + 4 * df["num_pageviews"]
        + 3 * df["trial_days_used"]
    ).clip(0, 100)
    return df, score.to_numpy()


# ---------------------------------------------------------------------------
# fraud-detect-v4: binary classification (fraud / clean)
# ---------------------------------------------------------------------------

_FRAUD_CATEGORIES = ["retail", "food", "travel", "digital", "services"]
_FRAUD_CATEGORY_P = [0.35, 0.25, 0.15, 0.15, 0.10]


def generate_fraud(state: State, n: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    if state == "baseline":
        amount = _RNG.lognormal(np.log(45), 1.0, n).clip(1, 5000)
        risk_score = _RNG.normal(0.15, 0.10, n).clip(0, 1)
    elif state == "warning":
        amount = _RNG.lognormal(np.log(65), 1.0, n).clip(1, 5000)
        risk_score = _RNG.normal(0.18, 0.11, n).clip(0, 1)
    else:  # critical
        amount = _RNG.lognormal(np.log(220), 1.2, n).clip(1, 5000)
        risk_score = _RNG.normal(0.42, 0.18, n).clip(0, 1)

    category = _RNG.choice(_FRAUD_CATEGORIES, size=n, p=_FRAUD_CATEGORY_P)
    card_age = _RNG.normal(380, 220, n).clip(1, 3000)
    hour = _RNG.integers(0, 24, n).astype(float)

    df = pd.DataFrame({
        "transaction_amount": amount.round(2),
        "merchant_category": category,
        "card_age_days": card_age.round(0),
        "country_risk_score": risk_score.round(3),
        "time_of_day_hour": hour,
    })

    # Toy fraud rule: large amount + high country risk + new card = flag
    flag = (
        (df["transaction_amount"] > 200)
        & (df["country_risk_score"] > 0.3)
        & (df["card_age_days"] < 90)
    )
    return df, flag.astype(int).to_numpy()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "churn-v3":        generate_churn,
    "lead-score-v2":   generate_lead,
    "fraud-detect-v4": generate_fraud,
}

MODEL_IDS = list(GENERATORS.keys())
