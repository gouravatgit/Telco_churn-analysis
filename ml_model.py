"""
ml_model.py
-----------
Train a Random Forest classifier to predict churn probability.
Returns per-customer churn risk scores and a feature importance table.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────

CAT_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
    "ProductTier", "TenureBand", "Region",
]

NUM_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
]


def _build_feature_matrix(df: pd.DataFrame):
    """One-hot encode categoricals + numeric features → X, y."""
    X_cat = pd.get_dummies(df[CAT_FEATURES], drop_first=True)
    X_num = df[NUM_FEATURES].copy()
    X = pd.concat([X_num.reset_index(drop=True),
                   X_cat.reset_index(drop=True)], axis=1)
    y = df["ChurnFlag"].reset_index(drop=True)
    return X, y


# ──────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Train a Gradient Boosting classifier.

    Returns
    -------
    model      : fitted estimator
    X_test     : held-out features
    y_test     : held-out labels
    metrics    : dict with auc, cv_auc, classification_report
    feat_imp   : DataFrame of feature importances
    """
    X, y = _build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

    metrics = {
        "auc": round(auc, 4),
        "cv_auc_mean": round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    feat_imp = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    return model, X, y, X_test, y_test, metrics, feat_imp


# ──────────────────────────────────────────────
# Score full dataset
# ──────────────────────────────────────────────

def score_customers(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Append ChurnProbability and RiskTier columns to df.

    Risk tiers
    ----------
    High   : probability >= 0.60
    Medium : 0.35 <= probability < 0.60
    Low    : probability < 0.35
    """
    X, _ = _build_feature_matrix(df)
    proba = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["ChurnProbability"] = proba.round(4)
    result["RiskTier"] = pd.cut(
        result["ChurnProbability"],
        bins=[-0.001, 0.35, 0.60, 1.001],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return result
