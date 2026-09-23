"""
data_loader.py
--------------
Handles loading, cleaning, and feature engineering for the
Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 1. Load
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    df = pd.read_csv(filepath)
    return df


# ─────────────────────────────────────────────
# 2. Inspect
# ─────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> dict:
    """Return a summary dict of shape, dtypes, missing values, duplicates."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    # TotalCharges can come in as object with spaces
    tc_blanks = 0
    if df["TotalCharges"].dtype == object:
        tc_blanks = (df["TotalCharges"].str.strip() == "").sum()

    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_counts": missing[missing > 0].to_dict(),
        "missing_pct": missing_pct[missing_pct > 0].to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "totalcharges_blank_spaces": int(tc_blanks),
    }


# ─────────────────────────────────────────────
# 3. Clean
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps and return a clean DataFrame.

    Steps
    -----
    1. Strip whitespace from object columns.
    2. Convert TotalCharges to numeric (blank → NaN → fill with MonthlyCharges).
    3. Drop exact duplicates.
    4. Standardise Churn to binary int (1 = churned, 0 = retained).
    5. Derive a synthetic 'Region' and 'ProductTier' column for
       dashboard filtering (dataset does not contain these; we infer them).
    6. Derive a cohort tenure band.
    """
    df = df.copy()

    # 1. Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 2. Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_mask = df["TotalCharges"].isna()
    df.loc[blank_mask, "TotalCharges"] = df.loc[blank_mask, "MonthlyCharges"]

    # 3. Drop duplicates
    df = df.drop_duplicates()

    # 4. Binary churn flag
    df["ChurnFlag"] = (df["Churn"].str.lower() == "yes").astype(int)

    # 5a. Synthetic Region  (deterministic hash so it's reproducible)
    regions = ["North", "South", "East", "West"]
    df["Region"] = df["customerID"].apply(
        lambda cid: regions[hash(cid[:4]) % 4]
    )

    # 5b. Product Tier derived from MonthlyCharges
    bins   = [0, 35, 70, 105, np.inf]
    labels = ["Basic", "Standard", "Premium", "Enterprise"]
    df["ProductTier"] = pd.cut(
        df["MonthlyCharges"], bins=bins, labels=labels, right=True
    ).astype(str)

    # 6. Tenure band
    df["TenureBand"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
        right=True,
    ).astype(str)

    return df


# ─────────────────────────────────────────────
# 4. Full pipeline
# ─────────────────────────────────────────────

def get_clean_data(filepath: str):
    """Convenience: load → inspect → clean → return (df_clean, report)."""
    raw = load_data(filepath)
    report = inspect_data(raw)
    clean = clean_data(raw)
    return clean, report
