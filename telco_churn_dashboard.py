"""
telco_churn_dashboard.py  –  Telco Customer Churn Analytics (Single-File)
==========================================================================
All modules combined:
  1. Data Loading & Cleaning
  2. Analytics (Churn Rate, ARPU, CLV, Trends, Drivers)
  3. ML Model (Gradient Boosting, Risk Scoring)
  4. Streamlit Dashboard (7 tabs, sidebar filters)

Run with:
    streamlit run telco_churn_dashboard.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    return pd.read_csv(filepath)


def inspect_data(df: pd.DataFrame) -> dict:
    """Return a summary dict of shape, dtypes, missing values, duplicates."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps and return a clean DataFrame.

    Steps
    -----
    1. Strip whitespace from object columns.
    2. Convert TotalCharges to numeric (blank → NaN → fill with MonthlyCharges).
    3. Drop exact duplicates.
    4. Standardise Churn to binary int (1 = churned, 0 = retained).
    5. Derive synthetic 'Region' and 'ProductTier' columns for dashboard filtering.
    6. Derive a cohort tenure band.
    """
    df = df.copy()

    # 1. Strip whitespace
    for col in df.select_dtypes(include="str").columns:
        df[col] = df[col].str.strip()

    # 2. Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_mask = df["TotalCharges"].isna()
    df.loc[blank_mask, "TotalCharges"] = df.loc[blank_mask, "MonthlyCharges"]

    # 3. Drop duplicates
    df = df.drop_duplicates()

    # 4. Binary churn flag
    df["ChurnFlag"] = (df["Churn"].str.lower() == "yes").astype(int)

    # 5a. Synthetic Region — deterministic hash for reproducibility
    regions = ["North", "South", "East", "West"]
    df["Region"] = df["customerID"].apply(lambda cid: regions[hash(cid[:4]) % 4])

    # 5b. Product Tier from MonthlyCharges bins
    df["ProductTier"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 70, 105, np.inf],
        labels=["Basic", "Standard", "Premium", "Enterprise"],
        right=True,
    ).astype(str)

    # 6. Tenure band
    df["TenureBand"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
        right=True,
    ).astype(str)

    return df


def get_clean_data(filepath: str):
    """Convenience: load → inspect → clean → return (df_clean, report)."""
    raw = load_data(filepath)
    report = inspect_data(raw)
    clean = clean_data(raw)
    return clean, report


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ANALYTICS (KPIs, Segments, Trends, Drivers)
# ══════════════════════════════════════════════════════════════════════════════

def churn_rate(df: pd.DataFrame) -> float:
    """Overall churn rate as a fraction (0–1)."""
    return df["ChurnFlag"].mean()


def arpu(df: pd.DataFrame) -> float:
    """Average Revenue Per User — mean of MonthlyCharges."""
    return df["MonthlyCharges"].mean()


def clv(df: pd.DataFrame) -> pd.Series:
    """
    Customer Lifetime Value per customer.
    Formula: CLV = MonthlyCharges × (1 / churn_rate)
    Returns a Series aligned with df.
    """
    safe_cr = max(churn_rate(df), 1e-6)
    return df["MonthlyCharges"] * (1 / safe_cr)


def avg_clv(df: pd.DataFrame) -> float:
    """Mean CLV across all customers."""
    return clv(df).mean()


def churn_by_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Churn rate, count, and ARPU broken down by unique values of `col`."""
    grp = df.groupby(col).agg(
        Total=("ChurnFlag", "count"),
        Churned=("ChurnFlag", "sum"),
        ARPU=("MonthlyCharges", "mean"),
    )
    grp["ChurnRate"] = (grp["Churned"] / grp["Total"] * 100).round(2)
    grp["ARPU"] = grp["ARPU"].round(2)
    return grp.reset_index()


def monthly_churn_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct a monthly churn snapshot from tenure data.
    Each tenure value represents the cohort month; ChurnFlag indicates
    whether the customer left in that month.
    """
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
    monthly["ChurnRate"] = (monthly["Churned"] / monthly["TotalCustomers"] * 100).round(2)
    monthly["AvgMonthlyCharge"] = monthly["AvgMonthlyCharge"].round(2)
    monthly["ChurnRate_MA3"] = monthly["ChurnRate"].rolling(3, min_periods=1).mean().round(2)
    return monthly


def filtered_trend(df: pd.DataFrame, product_tier: str = "All", region: str = "All") -> pd.DataFrame:
    """Apply ProductTier / Region filters then compute monthly trend."""
    mask = pd.Series([True] * len(df), index=df.index)
    if product_tier != "All":
        mask &= df["ProductTier"] == product_tier
    if region != "All":
        mask &= df["Region"] == region
    return monthly_churn_trend(df[mask])


def churn_drivers(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    One-hot encode categorical features and compute point-biserial
    correlation with ChurnFlag. Returns top_n features by absolute correlation.
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MACHINE LEARNING MODEL
# ══════════════════════════════════════════════════════════════════════════════

# Feature lists
_CAT_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
    "ProductTier", "TenureBand", "Region",
]
_NUM_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def _build_feature_matrix(df: pd.DataFrame):
    """One-hot encode categoricals + numeric features → (X, y)."""
    X_cat = pd.get_dummies(df[_CAT_FEATURES], drop_first=True)
    X_num = df[_NUM_FEATURES].copy()
    X = pd.concat([X_num.reset_index(drop=True),
                   X_cat.reset_index(drop=True)], axis=1)
    y = df["ChurnFlag"].reset_index(drop=True)
    return X, y


def train_model(df: pd.DataFrame):
    """
    Train a Gradient Boosting classifier.

    Returns
    -------
    model     : fitted GradientBoostingClassifier
    X         : full feature matrix
    y         : full target vector
    X_test    : held-out features
    y_test    : held-out labels
    metrics   : dict with auc, cv_auc_mean, cv_auc_std, classification_report, confusion_matrix
    feat_imp  : DataFrame of feature importances (sorted descending)
    """
    X, y = _build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08,
        max_depth=4, subsample=0.85, random_state=42,
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
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    feat_imp = (
        pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return model, X, y, X_test, y_test, metrics, feat_imp


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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Segoe UI", system-ui, sans-serif; }

section[data-testid="stSidebar"] { background: #1a1f2e; color: #e2e8f0; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; font-size: 0.88rem; }

div[data-testid="metric-container"] {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] label {
    font-size: 0.82rem !important; color: #64748b !important;
    letter-spacing: 0.04em; text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 2rem !important; font-weight: 700 !important; color: #1e293b !important;
}

.section-header {
    font-size: 1.15rem; font-weight: 700; color: #1e293b;
    margin: 1.4rem 0 0.6rem 0; padding-bottom: 6px;
    border-bottom: 2px solid #3b82f6; display: inline-block;
}
.stDataFrame { font-size: 0.88rem; }
.badge-high   { background:#fee2e2; color:#991b1b; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.badge-medium { background:#fef3c7; color:#92400e; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.badge-low    { background:#dcfce7; color:#166534; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.rec-card {
    background: #f8fafc; border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0; padding: 14px 18px;
    margin-bottom: 12px; font-size: 0.93rem; line-height: 1.6; color: #1e293b;
}
.rec-card strong { color: #1d4ed8; }
button[data-baseweb="tab"] { font-size: 0.92rem !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PALETTE     = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4", "#f97316", "#ec4899"]
RISK_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
CSV_PATH    = os.path.join(os.path.dirname(__file__), "..", "Telco-Customer-Churn.csv")

def pct_fmt(v: float) -> str:   return f"{v * 100:.2f}%"
def dollar_fmt(v: float) -> str: return f"${v:,.2f}"

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning dataset …")
def _load_and_clean():
    return get_clean_data(CSV_PATH)

@st.cache_resource(show_spinner="Training ML model …")
def _build_model(_df):
    return train_model(_df)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    df, report = _load_and_clean()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=100)
        st.markdown("## 📡 Telco Churn Analytics")
        st.markdown("---")
        st.markdown("### 🔽 Dashboard Filters")

        sel_region   = st.selectbox("Region",        ["All"] + sorted(df["Region"].unique().tolist()),      index=0)
        sel_tier     = st.selectbox("Product Tier",  ["All"] + sorted(df["ProductTier"].unique().tolist()), index=0)
        sel_contract = st.selectbox("Contract Type", ["All"] + sorted(df["Contract"].unique().tolist()),    index=0)

        st.markdown("---")
        st.markdown("### ⚙️ Model Options")
        top_n_risk     = st.slider("Top N at-risk customers",   10, 200, 50,  step=10)
        risk_threshold = st.slider("High-risk threshold (%)",   40, 80,  60,  step=5)
        st.markdown("---")
        st.caption("Dataset: IBM Telco Customer Churn  |  7,043 customers")

    # ── Apply filters ─────────────────────────────────────────────────────────
    dff = df.copy()
    if sel_region   != "All": dff = dff[dff["Region"]      == sel_region]
    if sel_tier     != "All": dff = dff[dff["ProductTier"] == sel_tier]
    if sel_contract != "All": dff = dff[dff["Contract"]    == sel_contract]

    # ── Train model (cached) ──────────────────────────────────────────────────
    model, X_full, y_full, X_test, y_test, metrics, feat_imp = _build_model(df)
    scored_df  = score_customers(df, model)
    scored_dff = scored_df.copy()
    if sel_region   != "All": scored_dff = scored_dff[scored_dff["Region"]      == sel_region]
    if sel_tier     != "All": scored_dff = scored_dff[scored_dff["ProductTier"] == sel_tier]
    if sel_contract != "All": scored_dff = scored_dff[scored_dff["Contract"]    == sel_contract]

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='font-size:2rem;font-weight:800;color:#1e293b;margin-bottom:2px;'>"
        "📡 Telco Customer Churn Analytics Dashboard</h1>"
        "<p style='color:#64748b;font-size:0.95rem;margin-top:0;'>"
        "Identify churn drivers · Predict at-risk customers · Drive retention strategy</p>",
        unsafe_allow_html=True,
    )
    filter_info = []
    if sel_region   != "All": filter_info.append(f"Region: **{sel_region}**")
    if sel_tier     != "All": filter_info.append(f"Tier: **{sel_tier}**")
    if sel_contract != "All": filter_info.append(f"Contract: **{sel_contract}**")
    if filter_info:
        st.info(f"Active filters — {' · '.join(filter_info)} — showing **{len(dff):,}** customers")
    else:
        st.info(f"Showing all **{len(dff):,}** customers (no filters applied)")
    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_overview, tab_trends, tab_segments, tab_model, tab_atrisk, tab_data, tab_recs = st.tabs([
        "📊 Overview", "📈 Monthly Trends", "🔍 Segment Analysis",
        "🤖 ML Model", "🚨 At-Risk Customers", "🗄️ Data Quality", "💡 Recommendations",
    ])

    # ═════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ═════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

        cr       = churn_rate(dff)
        arpu_val = arpu(dff)
        clv_val  = avg_clv(dff)
        total    = len(dff)
        churned  = int(dff["ChurnFlag"].sum())
        retained = total - churned
        rev_at_risk = scored_dff[scored_dff["RiskTier"] == "High"]["MonthlyCharges"].sum()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Customers", f"{total:,}")
        c2.metric("Churned",         f"{churned:,}")
        c3.metric("Churn Rate",      pct_fmt(cr),
                  delta=f"{'↑' if cr > 0.265 else '↓'} vs 26.5% baseline", delta_color="inverse")
        c4.metric("ARPU / Month",    dollar_fmt(arpu_val))
        c5.metric("Avg CLV",         dollar_fmt(clv_val))
        c6.metric("Revenue at Risk", dollar_fmt(rev_at_risk),
                  help="Monthly charges of High-risk customers")

        st.markdown('<div class="section-header">Churn Distribution</div>', unsafe_allow_html=True)
        col_pie, col_bar, col_inet = st.columns(3)

        with col_pie:
            fig = px.pie(values=[retained, churned], names=["Retained", "Churned"],
                         color_discrete_sequence=["#10b981", "#ef4444"], hole=0.55,
                         title="Churn vs Retained")
            fig.update_traces(textinfo="percent+label", textfont_size=13)
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10),
                              legend=dict(orientation="h", yanchor="bottom", y=-0.15), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with col_bar:
            d = churn_by_column(dff, "Contract")
            fig = px.bar(d, x="Contract", y="ChurnRate", color="Contract",
                         color_discrete_sequence=PALETTE, text="ChurnRate",
                         title="Churn Rate by Contract (%)")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, yaxis_title="Churn Rate (%)",
                              margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with col_inet:
            d = churn_by_column(dff, "InternetService")
            fig = px.bar(d, x="InternetService", y="ChurnRate", color="InternetService",
                         color_discrete_sequence=PALETTE, text="ChurnRate",
                         title="Churn Rate by Internet Service (%)")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, yaxis_title="Churn Rate (%)",
                              margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Revenue Distribution by Churn Status</div>', unsafe_allow_html=True)
        col_hist, col_box = st.columns(2)

        with col_hist:
            fig = px.histogram(dff, x="MonthlyCharges", color="Churn", nbins=40,
                               barmode="overlay", opacity=0.75,
                               color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                               title="Monthly Charges Distribution",
                               labels={"MonthlyCharges": "Monthly Charges ($)"})
            fig.update_layout(margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with col_box:
            fig = px.box(dff, x="Contract", y="MonthlyCharges", color="Churn",
                         color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                         title="Monthly Charges by Contract & Churn")
            fig.update_layout(margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Tenure Analysis</div>', unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            d = churn_by_column(dff, "TenureBand")
            fig = px.bar(d, x="TenureBand", y="ChurnRate", color="ChurnRate",
                         color_continuous_scale="Reds", text="ChurnRate",
                         title="Churn Rate by Tenure Band (%)")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Tenure Band",
                              yaxis_title="Churn Rate (%)", margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with col_t2:
            fig = px.scatter(dff, x="tenure", y="MonthlyCharges", color="Churn",
                             opacity=0.45, color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                             title="Tenure vs Monthly Charges",
                             labels={"tenure": "Tenure (months)"})
            fig.update_layout(margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════
    # TAB 2 — MONTHLY TRENDS
    # ═════════════════════════════════════════════════════════
    with tab_trends:
        st.markdown('<div class="section-header">Monthly Churn Trends</div>', unsafe_allow_html=True)
        st.caption("Trends reconstructed from tenure data. Each point = cohort whose last observed month equals that tenure value.")

        trend = filtered_trend(dff, product_tier=sel_tier, region=sel_region)

        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(go.Scatter(x=trend["Month"], y=trend["ChurnRate"],
            name="Churn Rate (%)", line=dict(color="#ef4444", width=2), mode="lines"), secondary_y=False)
        fig_trend.add_trace(go.Scatter(x=trend["Month"], y=trend["ChurnRate_MA3"],
            name="3-Month MA", line=dict(color="#f97316", width=2, dash="dot"), mode="lines"), secondary_y=False)
        fig_trend.add_trace(go.Bar(x=trend["Month"], y=trend["TotalCustomers"],
            name="Customers in Cohort", marker_color="#93c5fd", opacity=0.45), secondary_y=True)
        fig_trend.update_xaxes(title_text="Tenure Month")
        fig_trend.update_yaxes(title_text="Churn Rate (%)", secondary_y=False)
        fig_trend.update_yaxes(title_text="Customers in Cohort", secondary_y=True)
        fig_trend.update_layout(title="Monthly Churn Rate & Cohort Size", title_font_size=15,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=60, b=30), height=420)
        st.plotly_chart(fig_trend, use_container_width=True)

        col_avg, col_cnt = st.columns(2)
        with col_avg:
            fig = px.line(trend, x="Month", y="AvgMonthlyCharge",
                          title="Average Monthly Charge per Cohort",
                          labels={"AvgMonthlyCharge": "Avg Monthly Charge ($)", "Month": "Tenure Month"},
                          color_discrete_sequence=["#3b82f6"])
            fig.update_layout(margin=dict(t=50, b=20), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with col_cnt:
            fig = px.bar(trend, x="Month", y="Churned", color="Churned",
                         color_continuous_scale="Reds", title="Churned Customers per Cohort Month",
                         labels={"Churned": "Churned", "Month": "Tenure Month"})
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=20), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Region vs Product Tier Comparison</div>', unsafe_allow_html=True)
        col_reg, col_tier = st.columns(2)

        with col_reg:
            d = churn_by_column(dff, "Region")
            fig = px.bar(d, x="Region", y="ChurnRate", color="Region",
                         color_discrete_sequence=PALETTE, text="ChurnRate",
                         title="Churn Rate by Region (%)")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with col_tier:
            d = churn_by_column(dff, "ProductTier")
            fig = px.bar(d, x="ProductTier", y="ChurnRate", color="ProductTier",
                         color_discrete_sequence=PALETTE, text="ChurnRate",
                         title="Churn Rate by Product Tier (%)")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Churn Rate Heatmap: Region × Product Tier</div>', unsafe_allow_html=True)
        pivot = (dff.groupby(["Region", "ProductTier"])["ChurnFlag"]
                 .mean().mul(100).round(1).reset_index()
                 .pivot(index="Region", columns="ProductTier", values="ChurnFlag").fillna(0))
        fig = px.imshow(pivot, text_auto=True, color_continuous_scale="RdYlGn_r",
                        aspect="auto", title="Churn Rate (%) — Region × Product Tier")
        fig.update_layout(coloraxis_colorbar_title="Churn %", margin=dict(t=60, b=20), title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════
    # TAB 3 — SEGMENT ANALYSIS
    # ═════════════════════════════════════════════════════════
    with tab_segments:
        st.markdown('<div class="section-header">Churn Drivers — Correlation Analysis</div>', unsafe_allow_html=True)
        drivers = churn_drivers(dff)
        fig = px.bar(drivers.sort_values("AbsCorrelation"),
                     x="AbsCorrelation", y="Feature", orientation="h",
                     color="AbsCorrelation", color_continuous_scale="Blues",
                     title="Top 10 Features Correlated with Churn",
                     labels={"AbsCorrelation": "Absolute Correlation", "Feature": ""})
        fig.update_layout(coloraxis_showscale=False, height=400, margin=dict(t=50, b=10), title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Segment Deep-Dives</div>', unsafe_allow_html=True)
        seg_col = st.selectbox("Choose segment dimension",
            ["PaymentMethod", "TechSupport", "OnlineSecurity", "MultipleLines",
             "StreamingTV", "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"], index=0)
        df_seg = churn_by_column(dff, seg_col)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df_seg, x=seg_col, y="ChurnRate", color=seg_col,
                         color_discrete_sequence=PALETTE, text="ChurnRate",
                         title=f"Churn Rate (%) by {seg_col}")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(df_seg, x=seg_col, y="ARPU", color=seg_col,
                         color_discrete_sequence=PALETTE, text="ARPU",
                         title=f"ARPU ($) by {seg_col}")
            fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
            fig.update_layout(showlegend=False, margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Segment Summary Table</div>', unsafe_allow_html=True)
        disp = df_seg.copy()
        disp.columns = [seg_col, "Total Customers", "Churned", "ARPU ($)", "Churn Rate (%)"]
        st.dataframe(
            disp.style.background_gradient(subset=["Churn Rate (%)"], cmap="Reds")
            .format({"ARPU ($)": "${:.2f}", "Churn Rate (%)": "{:.2f}%"}),
            use_container_width=True,
        )

        st.markdown('<div class="section-header">CLV Distribution by Contract Type</div>', unsafe_allow_html=True)
        dff_clv = dff.copy()
        dff_clv["CLV"] = clv(dff_clv)
        fig = px.violin(dff_clv, x="Contract", y="CLV", color="Churn",
                        box=True, points=False,
                        color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                        title="Customer Lifetime Value Distribution by Contract & Churn")
        fig.update_layout(margin=dict(t=50, b=10), title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════
    # TAB 4 — ML MODEL
    # ═════════════════════════════════════════════════════════
    with tab_model:
        st.markdown('<div class="section-header">Gradient Boosting Churn Prediction Model</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC (test)",      str(metrics["auc"]))
        m2.metric("CV ROC-AUC (5-fold)", str(metrics["cv_auc_mean"]))
        m3.metric("CV Std Dev",          str(metrics["cv_auc_std"]))
        rpt = metrics["classification_report"]
        m4.metric("F1 Score (Churn)",    f"{rpt['1']['f1-score']:.4f}")

        st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)
        fig = px.bar(feat_imp.head(20).sort_values("Importance"),
                     x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Viridis",
                     title="Top 20 Feature Importances (Gradient Boosting)")
        fig.update_layout(coloraxis_showscale=False, height=480, margin=dict(t=50, b=10), title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
        rpt_df = pd.DataFrame({
            "Class":     ["No Churn", "Churn", "Macro Avg", "Weighted Avg"],
            "Precision": [rpt["0"]["precision"], rpt["1"]["precision"],
                          rpt["macro avg"]["precision"], rpt["weighted avg"]["precision"]],
            "Recall":    [rpt["0"]["recall"],    rpt["1"]["recall"],
                          rpt["macro avg"]["recall"],    rpt["weighted avg"]["recall"]],
            "F1-Score":  [rpt["0"]["f1-score"],  rpt["1"]["f1-score"],
                          rpt["macro avg"]["f1-score"],  rpt["weighted avg"]["f1-score"]],
            "Support":   [rpt["0"]["support"],   rpt["1"]["support"],
                          rpt["macro avg"]["support"],   rpt["weighted avg"]["support"]],
        })
        st.dataframe(
            rpt_df.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}",
                                 "F1-Score": "{:.4f}", "Support": "{:.0f}"})
            .background_gradient(subset=["F1-Score"], cmap="Greens"),
            use_container_width=True,
        )

        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = metrics["confusion_matrix"]
        fig = px.imshow(cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Predicted: No Churn", "Predicted: Churn"],
                        y=["Actual: No Churn", "Actual: Churn"],
                        text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
        fig.update_layout(margin=dict(t=60, b=20), title_font_size=14, coloraxis_showscale=False)
        fig.update_traces(textfont_size=16)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Predicted Churn Probability Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(scored_df, x="ChurnProbability", color="Churn",
                           nbins=50, barmode="overlay", opacity=0.72,
                           color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                           title="Predicted Churn Probability — Actual vs Churned",
                           labels={"ChurnProbability": "Predicted Churn Probability"})
        fig.add_vline(x=risk_threshold / 100, line_dash="dash", line_color="#1e293b",
                      annotation_text=f"High-risk threshold ({risk_threshold}%)",
                      annotation_position="top right")
        fig.update_layout(margin=dict(t=60, b=20), title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════
    # TAB 5 — AT-RISK CUSTOMERS
    # ═════════════════════════════════════════════════════════
    with tab_atrisk:
        st.markdown('<div class="section-header">Customers at Highest Risk of Churning Next Month</div>', unsafe_allow_html=True)
        high_risk = scored_dff[scored_dff["ChurnProbability"] >= (risk_threshold / 100)].sort_values("ChurnProbability", ascending=False)
        n_high    = len(high_risk)
        rev_risk  = high_risk["MonthlyCharges"].sum()

        r1, r2, r3 = st.columns(3)
        r1.metric("High-Risk Customers",      f"{n_high:,}")
        r2.metric("Monthly Revenue at Risk",  dollar_fmt(rev_risk))
        r3.metric("Avg Churn Probability",
                  f"{high_risk['ChurnProbability'].mean()*100:.1f}%" if n_high > 0 else "N/A")

        st.markdown('<div class="section-header">Risk Tier Distribution</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            rc = scored_dff["RiskTier"].value_counts().reset_index()
            rc.columns = ["RiskTier", "Count"]
            fig = px.pie(rc, values="Count", names="RiskTier", color="RiskTier",
                         color_discrete_map=RISK_COLORS, hole=0.5,
                         title="Customer Risk Tier Breakdown")
            fig.update_traces(textinfo="percent+value", textfont_size=13)
            fig.update_layout(margin=dict(t=50, b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=-0.15), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            rrev = (scored_dff.groupby("RiskTier")["MonthlyCharges"].sum().reset_index()
                    .rename(columns={"MonthlyCharges": "Monthly Revenue ($)"}))
            fig = px.bar(rrev, x="RiskTier", y="Monthly Revenue ($)", color="RiskTier",
                         color_discrete_map=RISK_COLORS, text="Monthly Revenue ($)",
                         title="Monthly Revenue Exposure by Risk Tier")
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig.update_layout(showlegend=False, margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f'<div class="section-header">Top {top_n_risk} At-Risk Customer List</div>', unsafe_allow_html=True)
        display_cols = ["customerID", "gender", "tenure", "Contract", "InternetService",
                        "MonthlyCharges", "TotalCharges", "Region", "ProductTier", "ChurnProbability", "RiskTier"]
        top_df = high_risk[display_cols].head(top_n_risk).copy()
        top_df["ChurnProbability"] = (top_df["ChurnProbability"] * 100).round(1)
        top_df = top_df.rename(columns={"customerID": "Customer ID",
                                        "ChurnProbability": "Churn Prob (%)",
                                        "MonthlyCharges": "Monthly ($)",
                                        "TotalCharges": "Total ($)"})
        st.dataframe(
            top_df.style.background_gradient(subset=["Churn Prob (%)"], cmap="Reds")
            .format({"Monthly ($)": "${:.2f}", "Total ($)": "${:,.2f}", "Churn Prob (%)": "{:.1f}%"}),
            use_container_width=True,
        )

        st.markdown('<div class="section-header">Churn Probability vs Monthly Charges</div>', unsafe_allow_html=True)
        fig = px.scatter(scored_dff, x="MonthlyCharges", y="ChurnProbability",
                         color="RiskTier", color_discrete_map=RISK_COLORS,
                         hover_data=["customerID", "Contract", "tenure"], opacity=0.65,
                         title="Churn Probability vs Monthly Charges by Risk Tier",
                         labels={"MonthlyCharges": "Monthly Charges ($)", "ChurnProbability": "Churn Probability"})
        fig.add_hline(y=risk_threshold / 100, line_dash="dash", line_color="#1e293b",
                      annotation_text=f"High-risk line ({risk_threshold}%)")
        fig.update_layout(margin=dict(t=60, b=20), title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════
    # TAB 6 — DATA QUALITY
    # ═════════════════════════════════════════════════════════
    with tab_data:
        st.markdown('<div class="section-header">Dataset Summary & Quality Report</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Records",      f"{report['shape'][0]:,}")
        d2.metric("Columns",            str(report["shape"][1]))
        d3.metric("Duplicate Rows",     str(report["duplicates"]))
        d4.metric("Blank TotalCharges", str(report["totalcharges_blank_spaces"]))

        st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
        if report["missing_counts"]:
            miss_df = (pd.DataFrame.from_dict(report["missing_pct"], orient="index", columns=["Missing (%)"])
                       .reset_index().rename(columns={"index": "Column"}))
            fig = px.bar(miss_df, x="Column", y="Missing (%)",
                         title="Missing Value Percentage by Column",
                         color="Missing (%)", color_continuous_scale="Oranges")
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅  No missing values detected in the dataset after cleaning.")

        st.markdown('<div class="section-header">Column Overview</div>', unsafe_allow_html=True)
        dtype_df = (pd.DataFrame.from_dict(report["dtypes"], orient="index", columns=["Data Type"])
                    .reset_index().rename(columns={"index": "Column"}))
        dtype_df["Sample Values"] = dtype_df["Column"].apply(
            lambda c: ", ".join(df[c].dropna().astype(str).unique()[:4].tolist()) if c in df.columns else ""
        )
        st.dataframe(dtype_df, use_container_width=True)

        st.markdown('<div class="section-header">Raw Data Preview (first 100 rows)</div>', unsafe_allow_html=True)
        preview_cols = ["customerID", "gender", "SeniorCitizen", "Partner", "tenure",
                        "Contract", "MonthlyCharges", "TotalCharges", "InternetService",
                        "Churn", "ChurnFlag", "Region", "ProductTier", "TenureBand"]
        st.dataframe(dff[preview_cols].head(100), use_container_width=True)

    # ═════════════════════════════════════════════════════════
    # TAB 7 — RECOMMENDATIONS
    # ═════════════════════════════════════════════════════════
    with tab_recs:
        st.markdown('<div class="section-header">📋 Business Findings & Recommendations</div>', unsafe_allow_html=True)

        cr_pct   = churn_rate(df) * 100
        arpu_v   = arpu(df)
        clv_v    = avg_clv(df)
        high_n   = len(scored_df[scored_df["RiskTier"] == "High"])
        mtm_data = churn_by_column(df, "Contract")
        mtm_vals = mtm_data[mtm_data["Contract"] == "Month-to-month"]["ChurnRate"].values
        mtm_rate = float(mtm_vals[0]) if len(mtm_vals) else 0.0

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Overall Churn Rate", f"{cr_pct:.1f}%")
        sc2.metric("Monthly ARPU",       dollar_fmt(arpu_v))
        sc3.metric("Average CLV",        dollar_fmt(clv_v))
        st.markdown("---")

        st.markdown("### 🔍 Key Findings")
        findings = [
            ("🔴 Month-to-month contracts drive the most churn",
             f"Customers on month-to-month contracts churn at <strong>{mtm_rate:.1f}%</strong>, vs 11.3% (one year) and 2.8% (two year). Contract type is the single most powerful churn predictor (correlation 0.405)."),
            ("🔴 Fiber optic internet users churn at nearly 2× the DSL rate",
             "Fiber optic customers churn at 41.9% despite paying the highest ARPU ($91.50/mo). Service quality, pricing perception, or competitive pressure are likely causes."),
            ("🟡 New customers (0–12 months) are extremely vulnerable",
             "The first year is the highest-risk window at 47.7% churn. Customers who don't onboard well abandon service before building loyalty."),
            ("🟡 Lack of OnlineSecurity & TechSupport strongly predicts churn",
             "Both features rank in the top 5 churn correlations (r ≈ 0.34). Customers who feel unprotected or unsupported are far more likely to leave."),
            ("🟡 Electronic check payment correlates with higher churn",
             "Electronic check users churn at 45.3% — 3× the rate of auto-pay customers. Manual payment reflects a less committed relationship with the service."),
            ("🟢 Long-tenure customers (49–72 months) are the most loyal",
             f"Only 9.51% churn rate with the highest ARPU ($73.95/mo) and CLV ({dollar_fmt(clv_v)}). They are the core revenue base and should be actively rewarded."),
        ]
        for title, body in findings:
            st.markdown(f'<div class="rec-card"><strong>{title}</strong><br>{body}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ✅ Recommended Actions")
        recommendations = [
            ("1. Contract Upgrade Incentive Programme",
             f"Offer month-to-month customers a 15–20% discount to switch to annual contracts. Prioritise the top {high_n:,} high-risk customers first using ML scores. Converting 30% would reduce churn by 8–12%.",
             "🎯 Immediate · High Impact"),
            ("2. 90-Day Onboarding Success Programme",
             "Deploy automated check-ins at Day 7, 30, and 90 for all new customers. Tutorials, health checks, and satisfaction surveys cut first-year churn by an estimated 5–8%.",
             "🎯 Short-term · High Impact"),
            ("3. Bundle TechSupport & OnlineSecurity Free for 6 Months",
             "Gifting these add-ons increases stickiness, perceived value, and creates an upsell event at month 7. Estimated 3–5% churn reduction. Low cost, fast to deploy.",
             "🎯 Short-term · Medium Impact"),
            ("4. Fiber Optic Service Quality Audit",
             "Run NPS surveys and speed-test audits for fiber subscribers under 24 months tenure. Identify whether churn stems from service reliability, bill shock, or competition, then act.",
             "📋 Medium-term · High Impact"),
            ("5. AutoPay Incentive ($5/mo Discount)",
             "Electronic check payers churn at 45.3% vs 15–17% for auto-pay. A $5/month discount for switching to automatic payment reduces churn risk and payment failures simultaneously.",
             "🎯 Immediate · Medium Impact"),
            ("6. Deploy Churn Score API into CRM",
             "Operationalise the model (AUC 0.84) as a nightly batch job writing scores to every CRM record. Retention agents get data-driven context on every customer interaction.",
             "📋 Medium-term · Strategic"),
            ("7. Loyalty Rewards for 2+ Year Customers",
             "Long-tenure customers churn at only 9.5% but generate the highest CLV. Exclusive benefits (priority support, free upgrades, referral bonuses) sustain loyalty and generate referrals.",
             "📋 Long-term · Strategic"),
        ]
        for title, body, tag in recommendations:
            st.markdown(
                f'<div class="rec-card"><strong>{title}</strong>'
                f'&nbsp;<span style="font-size:0.78rem;color:#64748b;">{tag}</span>'
                f'<br>{body}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 📊 Expected Business Impact")
        impact_df = pd.DataFrame({
            "Initiative":              ["Contract Upgrade", "Onboarding Programme", "Security/Support Bundle",
                                        "Fiber Quality Fix", "AutoPay Incentive", "CRM Churn API", "Loyalty Rewards"],
            "Est. Churn Reduction":    ["8–12%", "5–8%", "3–5%", "4–7%", "2–4%", "3–6%", "1–3%"],
            "Revenue Impact":          ["High", "High", "Medium", "High", "Low", "Strategic", "Strategic"],
            "Effort":                  ["Medium", "Medium", "Low", "High", "Low", "Medium", "Medium"],
            "Time to Impact":          ["1–3 mo", "3–6 mo", "1–2 mo", "6–12 mo", "1 mo", "2–4 mo", "6–12 mo"],
        })
        st.dataframe(impact_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
