"""
analytics.py
------------
Core KPI calculations:
  - Overall churn rate
  - ARPU  (Average Revenue Per User)
  - CLV   (Customer Lifetime Value)
  - Churn by segment / dimension
  - Monthly churn trend simulation
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# 1. Top-level KPIs
# ──────────────────────────────────────────────

def churn_rate(df: pd.DataFrame) -> float:
    """Overall churn rate as a fraction (0-1)."""
    return df["ChurnFlag"].mean()


def arpu(df: pd.DataFrame) -> float:
    """
    Average Revenue Per User.
    Uses MonthlyCharges averaged over all active + churned customers.
    """
    return df["MonthlyCharges"].mean()


def clv(df: pd.DataFrame) -> pd.Series:
    """
    Customer Lifetime Value per customer.

    Simple formula:
        CLV = MonthlyCharges × (1 / monthly_churn_rate)
    where monthly_churn_rate is derived from the customer's own tenure
    and whether they churned.

    Returns a Series indexed like df with CLV values.
    """
    monthly_cr = churn_rate(df)  # scalar
    # Avoid division-by-zero if churn rate is 0
    safe_cr = max(monthly_cr, 1e-6)
    return df["MonthlyCharges"] * (1 / safe_cr)


def avg_clv(df: pd.DataFrame) -> float:
    """Mean CLV across all customers."""
    return clv(df).mean()


# ──────────────────────────────────────────────
# 2. Segment-level breakdowns
# ──────────────────────────────────────────────

def churn_by_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Returns churn rate, count, and ARPU broken down by unique values
    of `col`.
    """
    grp = df.groupby(col).agg(
        Total=("ChurnFlag", "count"),
        Churned=("ChurnFlag", "sum"),
        ARPU=("MonthlyCharges", "mean"),
    )
    grp["ChurnRate"] = (grp["Churned"] / grp["Total"] * 100).round(2)
    grp["ARPU"] = grp["ARPU"].round(2)
    return grp.reset_index()


# ──────────────────────────────────────────────
# 3. Monthly churn trend (simulated from tenure)
# ──────────────────────────────────────────────

def monthly_churn_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    The dataset has no explicit month column.  We reconstruct a
    monthly snapshot by treating each customer's `tenure` as the
    month in which they were last seen, and their `ChurnFlag` as
    whether they left in that month.

    Returns a DataFrame with columns [Month, TotalCustomers,
    Churned, ChurnRate, NewCustomers].
    """
    # Build a cohort view: one row per (tenure_month, churned)
    monthly = (
        df.groupby("tenure")
        .agg(
            TotalCustomers=("customerID", "count"),
            Churned=("ChurnFlag", "sum"),
            AvgMonthlyCharge=("MonthlyCharges", "mean"),
        )
        .reset_index()
        .rename(columns={"tenure": "Month"})
    )
    monthly["ChurnRate"] = (
        monthly["Churned"] / monthly["TotalCustomers"] * 100
    ).round(2)
    monthly["AvgMonthlyCharge"] = monthly["AvgMonthlyCharge"].round(2)

    # Rolling 3-month average for trend smoothing
    monthly["ChurnRate_MA3"] = (
        monthly["ChurnRate"].rolling(3, min_periods=1).mean().round(2)
    )
    return monthly


# ──────────────────────────────────────────────
# 4. Filtered trend helper
# ──────────────────────────────────────────────

def filtered_trend(
    df: pd.DataFrame,
    product_tier: str = "All",
    region: str = "All",
) -> pd.DataFrame:
    """Apply ProductTier / Region filters then compute monthly trend."""
    mask = pd.Series([True] * len(df), index=df.index)
    if product_tier != "All":
        mask &= df["ProductTier"] == product_tier
    if region != "All":
        mask &= df["Region"] == region
    return monthly_churn_trend(df[mask])


# ──────────────────────────────────────────────
# 5. Top churn risk drivers (simple correlation)
# ──────────────────────────────────────────────

def churn_drivers(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    One-hot encode categorical features and compute point-biserial
    correlation with ChurnFlag.  Returns top_n features sorted by
    absolute correlation descending.
    """
    cat_cols = [
        "Contract", "InternetService", "PaymentMethod",
        "TechSupport", "OnlineSecurity", "MultipleLines",
        "ProductTier", "TenureBand", "PaperlessBilling",
    ]
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

    dummies = pd.get_dummies(df[cat_cols], drop_first=False)
    feats = pd.concat([df[num_cols].reset_index(drop=True),
                       dummies.reset_index(drop=True)], axis=1)
    target = df["ChurnFlag"].reset_index(drop=True)

    corr = feats.corrwith(target).abs().sort_values(ascending=False)
    result = corr.head(top_n).reset_index()
    result.columns = ["Feature", "AbsCorrelation"]
    result["AbsCorrelation"] = result["AbsCorrelation"].round(4)
    return result
