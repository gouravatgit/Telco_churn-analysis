"""
app.py  –  Telco Customer Churn Analytics Dashboard
=====================================================
Run with:  streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Local modules ──────────────────────────────────────────────────────────────
from data_loader import get_clean_data
from analytics import (
    churn_rate, arpu, avg_clv, clv,
    churn_by_column, monthly_churn_trend,
    filtered_trend, churn_drivers,
)
from ml_model import train_model, score_customers

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS  –  clean white/dark card theme, readable fonts
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Global ─────────────────────────────────────── */
html, body, [class*="css"] { font-family: "Segoe UI", system-ui, sans-serif; }

/* ── Sidebar ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #1a1f2e;
    color: #e2e8f0;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #cbd5e1 !important;
    font-size: 0.88rem;
}

/* ── KPI metric cards ────────────────────────────── */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] label {
    font-size: 0.82rem !important;
    color: #64748b !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
}

/* ── Section headers ─────────────────────────────── */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e293b;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #3b82f6;
    display: inline-block;
}

/* ── Table ───────────────────────────────────────── */
.stDataFrame { font-size: 0.88rem; }

/* ── Risk badges ─────────────────────────────────── */
.badge-high   { background:#fee2e2; color:#991b1b; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.badge-medium { background:#fef3c7; color:#92400e; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.badge-low    { background:#dcfce7; color:#166534; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }

/* ── Recommendation cards ────────────────────────── */
.rec-card {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-size: 0.93rem;
    line-height: 1.6;
    color: #1e293b;
}
.rec-card strong { color: #1d4ed8; }

/* ── Tab styling ─────────────────────────────────── */
button[data-baseweb="tab"] {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (cached)
# ══════════════════════════════════════════════════════════════════════════════

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "Telco-Customer-Churn.csv")

@st.cache_data(show_spinner="Loading & cleaning dataset …")
def load_and_clean():
    return get_clean_data(CSV_PATH)


@st.cache_resource(show_spinner="Training ML model …")
def build_model(df_hash: int):
    """Cache the trained model so it doesn't retrain on every interaction."""
    return train_model(_df_global)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: colour palette
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b",
           "#8b5cf6", "#06b6d4", "#f97316", "#ec4899"]

RISK_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}


def pct_fmt(v: float) -> str:
    return f"{v * 100:.2f}%"


def dollar_fmt(v: float) -> str:
    return f"${v:,.2f}"


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _df_global

    # ── Load data ─────────────────────────────────────────────────────────────
    df, report = load_and_clean()
    _df_global = df  # needed by cached model builder

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
            width=100,
        )
        st.markdown("## 📡 Telco Churn Analytics")
        st.markdown("---")

        st.markdown("### 🔽 Dashboard Filters")

        all_regions = ["All"] + sorted(df["Region"].unique().tolist())
        all_tiers   = ["All"] + sorted(df["ProductTier"].unique().tolist())
        all_contracts = ["All"] + sorted(df["Contract"].unique().tolist())

        sel_region   = st.selectbox("Region",       all_regions,   index=0)
        sel_tier     = st.selectbox("Product Tier", all_tiers,     index=0)
        sel_contract = st.selectbox("Contract Type", all_contracts, index=0)

        st.markdown("---")
        st.markdown("### ⚙️ Model Options")
        top_n_risk = st.slider("Top N at-risk customers", 10, 200, 50, step=10)
        risk_threshold = st.slider("High-risk threshold (%)", 40, 80, 60, step=5)

        st.markdown("---")
        st.caption("Dataset: IBM Telco Customer Churn  |  7,043 customers")

    # ── Apply sidebar filters ─────────────────────────────────────────────────
    dff = df.copy()
    if sel_region   != "All": dff = dff[dff["Region"]      == sel_region]
    if sel_tier     != "All": dff = dff[dff["ProductTier"] == sel_tier]
    if sel_contract != "All": dff = dff[dff["Contract"]    == sel_contract]

    n_filtered = len(dff)

    # ── Train model on full data ───────────────────────────────────────────────
    model, X_full, y_full, X_test, y_test, metrics, feat_imp = build_model(id(df))
    scored_df = score_customers(df, model)

    # Apply filters to scored df too
    scored_dff = scored_df.copy()
    if sel_region   != "All": scored_dff = scored_dff[scored_dff["Region"]      == sel_region]
    if sel_tier     != "All": scored_dff = scored_dff[scored_dff["ProductTier"] == sel_tier]
    if sel_contract != "All": scored_dff = scored_dff[scored_dff["Contract"]    == sel_contract]

    # ══════════════════════════════════════════════════════════════════════════
    # Page title
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown(
        "<h1 style='font-size:2rem;font-weight:800;color:#1e293b;margin-bottom:2px;'>"
        "📡 Telco Customer Churn Analytics Dashboard"
        "</h1>"
        "<p style='color:#64748b;font-size:0.95rem;margin-top:0;'>"
        "Identify churn drivers · Predict at-risk customers · Drive retention strategy"
        "</p>",
        unsafe_allow_html=True,
    )

    filter_info = []
    if sel_region   != "All": filter_info.append(f"Region: **{sel_region}**")
    if sel_tier     != "All": filter_info.append(f"Tier: **{sel_tier}**")
    if sel_contract != "All": filter_info.append(f"Contract: **{sel_contract}**")
    if filter_info:
        st.info(f"Active filters — {' · '.join(filter_info)} — showing **{n_filtered:,}** customers")
    else:
        st.info(f"Showing all **{n_filtered:,}** customers (no filters applied)")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════════

    tab_overview, tab_trends, tab_segments, tab_model, tab_atrisk, tab_data, tab_recs = st.tabs([
        "📊 Overview",
        "📈 Monthly Trends",
        "🔍 Segment Analysis",
        "🤖 ML Model",
        "🚨 At-Risk Customers",
        "🗄️ Data Quality",
        "💡 Recommendations",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – Overview KPIs
    # ══════════════════════════════════════════════════════════════════════════

    with tab_overview:
        st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

        cr       = churn_rate(dff)
        arpu_val = arpu(dff)
        clv_val  = avg_clv(dff)
        total    = len(dff)
        churned  = int(dff["ChurnFlag"].sum())
        retained = total - churned
        revenue_at_risk = (
            scored_dff[scored_dff["RiskTier"] == "High"]["MonthlyCharges"].sum()
        )

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Customers",    f"{total:,}")
        c2.metric("Churned",            f"{churned:,}")
        c3.metric("Churn Rate",         pct_fmt(cr),
                  delta=f"{'↑' if cr > 0.265 else '↓'} vs 26.5% baseline",
                  delta_color="inverse")
        c4.metric("ARPU / Month",       dollar_fmt(arpu_val))
        c5.metric("Avg CLV",            dollar_fmt(clv_val))
        c6.metric("Revenue at Risk",    dollar_fmt(revenue_at_risk),
                  help="Monthly charges of High-risk customers")

        st.markdown('<div class="section-header">Churn Distribution</div>', unsafe_allow_html=True)

        col_pie, col_bar, col_contract = st.columns(3)

        # ── Pie: churned vs retained
        with col_pie:
            fig_pie = px.pie(
                values=[retained, churned],
                names=["Retained", "Churned"],
                color_discrete_sequence=["#10b981", "#ef4444"],
                hole=0.55,
                title="Churn vs Retained",
            )
            fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
            fig_pie.update_layout(
                margin=dict(t=50, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                title_font_size=14,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Bar: churn by contract
        with col_bar:
            df_contract = churn_by_column(dff, "Contract")
            fig_bar = px.bar(
                df_contract, x="Contract", y="ChurnRate",
                color="Contract",
                color_discrete_sequence=PALETTE,
                title="Churn Rate by Contract Type (%)",
                text="ChurnRate",
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(
                showlegend=False,
                yaxis_title="Churn Rate (%)",
                margin=dict(t=50, b=10),
                title_font_size=14,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Bar: churn by internet service
        with col_contract:
            df_internet = churn_by_column(dff, "InternetService")
            fig_inet = px.bar(
                df_internet, x="InternetService", y="ChurnRate",
                color="InternetService",
                color_discrete_sequence=PALETTE,
                title="Churn Rate by Internet Service (%)",
                text="ChurnRate",
            )
            fig_inet.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_inet.update_layout(
                showlegend=False,
                yaxis_title="Churn Rate (%)",
                margin=dict(t=50, b=10),
                title_font_size=14,
            )
            st.plotly_chart(fig_inet, use_container_width=True)

        # ── Monthly charges distribution by churn
        st.markdown('<div class="section-header">Revenue Distribution by Churn Status</div>', unsafe_allow_html=True)

        col_hist, col_box = st.columns(2)
        with col_hist:
            fig_hist = px.histogram(
                dff, x="MonthlyCharges", color="Churn",
                nbins=40,
                barmode="overlay",
                opacity=0.75,
                color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                title="Monthly Charges Distribution",
                labels={"MonthlyCharges": "Monthly Charges ($)"},
            )
            fig_hist.update_layout(margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_box:
            fig_box = px.box(
                dff, x="Contract", y="MonthlyCharges", color="Churn",
                color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                title="Monthly Charges by Contract & Churn",
            )
            fig_box.update_layout(margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig_box, use_container_width=True)

        # ── Tenure vs churn
        st.markdown('<div class="section-header">Tenure Analysis</div>', unsafe_allow_html=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            df_tenure = churn_by_column(dff, "TenureBand")
            fig_tenure = px.bar(
                df_tenure, x="TenureBand", y="ChurnRate",
                color="ChurnRate",
                color_continuous_scale="Reds",
                title="Churn Rate by Tenure Band (%)",
                text="ChurnRate",
            )
            fig_tenure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_tenure.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Tenure Band",
                yaxis_title="Churn Rate (%)",
                margin=dict(t=50, b=10),
                title_font_size=14,
            )
            st.plotly_chart(fig_tenure, use_container_width=True)

        with col_t2:
            fig_scatter = px.scatter(
                dff, x="tenure", y="MonthlyCharges",
                color="Churn",
                opacity=0.45,
                color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                title="Tenure vs Monthly Charges",
                labels={"tenure": "Tenure (months)"},
            )
            fig_scatter.update_layout(margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig_scatter, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – Monthly Trends
    # ══════════════════════════════════════════════════════════════════════════

    with tab_trends:
        st.markdown('<div class="section-header">Monthly Churn Trends</div>', unsafe_allow_html=True)
        st.caption(
            "Trends are reconstructed from customer tenure data. Each point represents "
            "the cohort of customers whose last observed month equals that tenure value."
        )

        trend = filtered_trend(dff, product_tier=sel_tier, region=sel_region)

        # ── Dual-axis: churn rate + customer count
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])

        fig_trend.add_trace(
            go.Scatter(
                x=trend["Month"], y=trend["ChurnRate"],
                name="Churn Rate (%)",
                line=dict(color="#ef4444", width=2),
                mode="lines",
            ),
            secondary_y=False,
        )
        fig_trend.add_trace(
            go.Scatter(
                x=trend["Month"], y=trend["ChurnRate_MA3"],
                name="3-Month MA",
                line=dict(color="#f97316", width=2, dash="dot"),
                mode="lines",
            ),
            secondary_y=False,
        )
        fig_trend.add_trace(
            go.Bar(
                x=trend["Month"], y=trend["TotalCustomers"],
                name="Customers in Cohort",
                marker_color="#93c5fd",
                opacity=0.45,
            ),
            secondary_y=True,
        )

        fig_trend.update_xaxes(title_text="Tenure Month")
        fig_trend.update_yaxes(title_text="Churn Rate (%)", secondary_y=False)
        fig_trend.update_yaxes(title_text="Customers in Cohort", secondary_y=True)
        fig_trend.update_layout(
            title="Monthly Churn Rate & Cohort Size",
            title_font_size=15,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=60, b=30),
            height=420,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── Average monthly charge trend
        col_avg, col_churn_cnt = st.columns(2)
        with col_avg:
            fig_charge = px.line(
                trend, x="Month", y="AvgMonthlyCharge",
                title="Average Monthly Charge per Cohort",
                labels={"AvgMonthlyCharge": "Avg Monthly Charge ($)", "Month": "Tenure Month"},
                color_discrete_sequence=["#3b82f6"],
            )
            fig_charge.update_layout(margin=dict(t=50, b=20), title_font_size=14)
            st.plotly_chart(fig_charge, use_container_width=True)

        with col_churn_cnt:
            fig_cnt = px.bar(
                trend, x="Month", y="Churned",
                title="Churned Customers per Cohort Month",
                color="Churned",
                color_continuous_scale="Reds",
                labels={"Churned": "Churned", "Month": "Tenure Month"},
            )
            fig_cnt.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=50, b=20),
                title_font_size=14,
            )
            st.plotly_chart(fig_cnt, use_container_width=True)

        # ── Region comparison
        st.markdown('<div class="section-header">Region vs Product Tier Comparison</div>', unsafe_allow_html=True)

        col_reg, col_tier_cmp = st.columns(2)
        with col_reg:
            df_reg = churn_by_column(dff, "Region")
            fig_reg = px.bar(
                df_reg, x="Region", y="ChurnRate",
                color="Region",
                color_discrete_sequence=PALETTE,
                text="ChurnRate",
                title="Churn Rate by Region (%)",
            )
            fig_reg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_reg.update_layout(showlegend=False, margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig_reg, use_container_width=True)

        with col_tier_cmp:
            df_tier = churn_by_column(dff, "ProductTier")
            fig_tier = px.bar(
                df_tier, x="ProductTier", y="ChurnRate",
                color="ProductTier",
                color_discrete_sequence=PALETTE,
                text="ChurnRate",
                title="Churn Rate by Product Tier (%)",
            )
            fig_tier.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_tier.update_layout(showlegend=False, margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig_tier, use_container_width=True)

        # ── Heatmap: region × product tier churn rate
        st.markdown('<div class="section-header">Churn Rate Heatmap: Region × Product Tier</div>', unsafe_allow_html=True)
        pivot = (
            dff.groupby(["Region", "ProductTier"])["ChurnFlag"]
            .mean()
            .mul(100)
            .round(1)
            .reset_index()
            .pivot(index="Region", columns="ProductTier", values="ChurnFlag")
            .fillna(0)
        )
        fig_heat = px.imshow(
            pivot,
            text_auto=True,
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            title="Churn Rate (%) — Region × Product Tier",
        )
        fig_heat.update_layout(
            coloraxis_colorbar_title="Churn %",
            margin=dict(t=60, b=20),
            title_font_size=14,
        )
        st.plotly_chart(fig_heat, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – Segment Analysis
    # ══════════════════════════════════════════════════════════════════════════

    with tab_segments:
        st.markdown('<div class="section-header">Churn Drivers — Correlation Analysis</div>', unsafe_allow_html=True)

        drivers = churn_drivers(dff)
        fig_drv = px.bar(
            drivers.sort_values("AbsCorrelation"),
            x="AbsCorrelation", y="Feature",
            orientation="h",
            color="AbsCorrelation",
            color_continuous_scale="Blues",
            title="Top 10 Features Correlated with Churn",
            labels={"AbsCorrelation": "Absolute Correlation", "Feature": ""},
        )
        fig_drv.update_layout(
            coloraxis_showscale=False,
            height=400,
            margin=dict(t=50, b=10),
            title_font_size=14,
        )
        st.plotly_chart(fig_drv, use_container_width=True)

        # ── Side-by-side segment charts
        st.markdown('<div class="section-header">Segment Deep-Dives</div>', unsafe_allow_html=True)

        seg_col = st.selectbox(
            "Choose segment dimension",
            ["PaymentMethod", "TechSupport", "OnlineSecurity",
             "MultipleLines", "StreamingTV", "SeniorCitizen",
             "Partner", "Dependents", "PaperlessBilling"],
            index=0,
        )

        df_seg = churn_by_column(dff, seg_col)

        c_seg1, c_seg2 = st.columns(2)
        with c_seg1:
            fig_seg_cr = px.bar(
                df_seg, x=seg_col, y="ChurnRate",
                color=seg_col,
                color_discrete_sequence=PALETTE,
                text="ChurnRate",
                title=f"Churn Rate (%) by {seg_col}",
            )
            fig_seg_cr.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_seg_cr.update_layout(showlegend=False, margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig_seg_cr, use_container_width=True)

        with c_seg2:
            fig_seg_arpu = px.bar(
                df_seg, x=seg_col, y="ARPU",
                color=seg_col,
                color_discrete_sequence=PALETTE,
                text="ARPU",
                title=f"ARPU ($) by {seg_col}",
            )
            fig_seg_arpu.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
            fig_seg_arpu.update_layout(showlegend=False, margin=dict(t=50, b=10), title_font_size=14)
            st.plotly_chart(fig_seg_arpu, use_container_width=True)

        # Full segment table
        st.markdown('<div class="section-header">Segment Summary Table</div>', unsafe_allow_html=True)
        display_df = df_seg.copy()
        display_df.columns = [seg_col, "Total Customers", "Churned", "ARPU ($)", "Churn Rate (%)"]
        st.dataframe(
            display_df.style.background_gradient(
                subset=["Churn Rate (%)"], cmap="Reds"
            ).format({"ARPU ($)": "${:.2f}", "Churn Rate (%)": "{:.2f}%"}),
            use_container_width=True,
        )

        # CLV distribution by segment
        st.markdown('<div class="section-header">CLV Distribution by Contract Type</div>', unsafe_allow_html=True)
        dff_clv = dff.copy()
        dff_clv["CLV"] = clv(dff_clv)
        fig_clv = px.violin(
            dff_clv, x="Contract", y="CLV", color="Churn",
            box=True, points=False,
            color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
            title="Customer Lifetime Value Distribution by Contract & Churn",
        )
        fig_clv.update_layout(margin=dict(t=50, b=10), title_font_size=14)
        st.plotly_chart(fig_clv, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – ML Model
    # ══════════════════════════════════════════════════════════════════════════

    with tab_model:
        st.markdown('<div class="section-header">Gradient Boosting Churn Prediction Model</div>', unsafe_allow_html=True)

        # KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC (test)",   str(metrics["auc"]))
        m2.metric("CV ROC-AUC (5-fold)", str(metrics["cv_auc_mean"]))
        m3.metric("CV Std Dev",       str(metrics["cv_auc_std"]))
        cr_report = metrics["classification_report"]
        m4.metric("F1 Score (Churn)", f"{cr_report['1']['f1-score']:.4f}")

        st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)

        fig_fi = px.bar(
            feat_imp.head(20).sort_values("Importance"),
            x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            title="Top 20 Feature Importances (Gradient Boosting)",
        )
        fig_fi.update_layout(
            coloraxis_showscale=False,
            height=480,
            margin=dict(t=50, b=10),
            title_font_size=14,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── Classification report
        st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)

        rpt = cr_report
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
            rpt_df.style.format({
                "Precision": "{:.4f}", "Recall": "{:.4f}",
                "F1-Score": "{:.4f}", "Support": "{:.0f}",
            }).background_gradient(subset=["F1-Score"], cmap="Greens"),
            use_container_width=True,
        )

        # ── Confusion matrix as heatmap
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)

        cm = metrics["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Predicted: No Churn", "Predicted: Churn"],
            y=["Actual: No Churn", "Actual: Churn"],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix",
        )
        fig_cm.update_layout(
            margin=dict(t=60, b=20),
            title_font_size=14,
            coloraxis_showscale=False,
        )
        fig_cm.update_traces(textfont_size=16)
        st.plotly_chart(fig_cm, use_container_width=True)

        # ── Churn probability distribution
        st.markdown('<div class="section-header">Predicted Churn Probability Distribution</div>', unsafe_allow_html=True)

        fig_prob = px.histogram(
            scored_df, x="ChurnProbability", color="Churn",
            nbins=50, barmode="overlay", opacity=0.72,
            color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
            title="Predicted Churn Probability — Actual vs Churned",
            labels={"ChurnProbability": "Predicted Churn Probability"},
        )
        fig_prob.add_vline(
            x=risk_threshold / 100,
            line_dash="dash", line_color="#1e293b",
            annotation_text=f"High-risk threshold ({risk_threshold}%)",
            annotation_position="top right",
        )
        fig_prob.update_layout(margin=dict(t=60, b=20), title_font_size=14)
        st.plotly_chart(fig_prob, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 – At-Risk Customers
    # ══════════════════════════════════════════════════════════════════════════

    with tab_atrisk:
        st.markdown('<div class="section-header">Customers at Highest Risk of Churning Next Month</div>', unsafe_allow_html=True)

        high_risk = scored_dff[
            scored_dff["ChurnProbability"] >= (risk_threshold / 100)
        ].sort_values("ChurnProbability", ascending=False)

        n_high = len(high_risk)
        rev_risk = high_risk["MonthlyCharges"].sum()

        r1, r2, r3 = st.columns(3)
        r1.metric("High-Risk Customers", f"{n_high:,}")
        r2.metric("Monthly Revenue at Risk", dollar_fmt(rev_risk))
        r3.metric("Avg Churn Probability", f"{high_risk['ChurnProbability'].mean()*100:.1f}%"
                  if n_high > 0 else "N/A")

        # ── Risk tier distribution
        st.markdown('<div class="section-header">Risk Tier Distribution</div>', unsafe_allow_html=True)

        c_tier1, c_tier2 = st.columns(2)

        with c_tier1:
            risk_counts = scored_dff["RiskTier"].value_counts().reset_index()
            risk_counts.columns = ["RiskTier", "Count"]
            fig_risk_pie = px.pie(
                risk_counts, values="Count", names="RiskTier",
                color="RiskTier",
                color_discrete_map=RISK_COLORS,
                hole=0.5,
                title="Customer Risk Tier Breakdown",
            )
            fig_risk_pie.update_traces(textinfo="percent+value", textfont_size=13)
            fig_risk_pie.update_layout(
                margin=dict(t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                title_font_size=14,
            )
            st.plotly_chart(fig_risk_pie, use_container_width=True)

        with c_tier2:
            risk_revenue = (
                scored_dff.groupby("RiskTier")["MonthlyCharges"]
                .sum().reset_index()
                .rename(columns={"MonthlyCharges": "Monthly Revenue ($)"})
            )
            fig_risk_rev = px.bar(
                risk_revenue, x="RiskTier", y="Monthly Revenue ($)",
                color="RiskTier",
                color_discrete_map=RISK_COLORS,
                title="Monthly Revenue Exposure by Risk Tier",
                text="Monthly Revenue ($)",
            )
            fig_risk_rev.update_traces(
                texttemplate="$%{text:,.0f}", textposition="outside"
            )
            fig_risk_rev.update_layout(
                showlegend=False,
                margin=dict(t=50, b=10),
                title_font_size=14,
            )
            st.plotly_chart(fig_risk_rev, use_container_width=True)

        # ── Top N at-risk table
        st.markdown(
            f'<div class="section-header">Top {top_n_risk} At-Risk Customer List</div>',
            unsafe_allow_html=True,
        )

        display_cols = [
            "customerID", "gender", "tenure", "Contract",
            "InternetService", "MonthlyCharges", "TotalCharges",
            "Region", "ProductTier", "ChurnProbability", "RiskTier",
        ]
        top_risk_df = high_risk[display_cols].head(top_n_risk).copy()
        top_risk_df["ChurnProbability"] = (top_risk_df["ChurnProbability"] * 100).round(1)
        top_risk_df = top_risk_df.rename(columns={
            "customerID": "Customer ID",
            "ChurnProbability": "Churn Prob (%)",
            "MonthlyCharges": "Monthly ($)",
            "TotalCharges": "Total ($)",
        })

        st.dataframe(
            top_risk_df.style
            .background_gradient(subset=["Churn Prob (%)"], cmap="Reds")
            .format({
                "Monthly ($)": "${:.2f}",
                "Total ($)": "${:,.2f}",
                "Churn Prob (%)": "{:.1f}%",
            }),
            use_container_width=True,
        )

        # ── Scatter: churn probability vs monthly charges
        st.markdown('<div class="section-header">Churn Probability vs Monthly Charges</div>', unsafe_allow_html=True)

        fig_scatter_risk = px.scatter(
            scored_dff,
            x="MonthlyCharges",
            y="ChurnProbability",
            color="RiskTier",
            color_discrete_map=RISK_COLORS,
            hover_data=["customerID", "Contract", "tenure"],
            opacity=0.65,
            title="Churn Probability vs Monthly Charges by Risk Tier",
            labels={
                "MonthlyCharges": "Monthly Charges ($)",
                "ChurnProbability": "Churn Probability",
            },
        )
        fig_scatter_risk.add_hline(
            y=risk_threshold / 100,
            line_dash="dash",
            line_color="#1e293b",
            annotation_text=f"High-risk line ({risk_threshold}%)",
        )
        fig_scatter_risk.update_layout(margin=dict(t=60, b=20), title_font_size=14)
        st.plotly_chart(fig_scatter_risk, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 – Data Quality
    # ══════════════════════════════════════════════════════════════════════════

    with tab_data:
        st.markdown('<div class="section-header">Dataset Summary & Quality Report</div>', unsafe_allow_html=True)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Records",   f"{report['shape'][0]:,}")
        d2.metric("Columns",         str(report["shape"][1]))
        d3.metric("Duplicate Rows",  str(report["duplicates"]))
        d4.metric("Blank TotalCharges", str(report["totalcharges_blank_spaces"]))

        # Missing values
        st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)

        if report["missing_counts"]:
            miss_df = pd.DataFrame.from_dict(
                report["missing_pct"], orient="index", columns=["Missing (%)"]
            ).reset_index().rename(columns={"index": "Column"})
            fig_miss = px.bar(
                miss_df, x="Column", y="Missing (%)",
                title="Missing Value Percentage by Column",
                color="Missing (%)", color_continuous_scale="Oranges",
            )
            fig_miss.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10))
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("✅  No missing values detected in the dataset after cleaning.")

        # Column dtypes
        st.markdown('<div class="section-header">Column Overview</div>', unsafe_allow_html=True)

        dtype_df = pd.DataFrame.from_dict(
            report["dtypes"], orient="index", columns=["Data Type"]
        ).reset_index().rename(columns={"index": "Column"})
        dtype_df["Sample Values"] = dtype_df["Column"].apply(
            lambda c: ", ".join(df[c].dropna().astype(str).unique()[:4].tolist())
            if c in df.columns else ""
        )
        st.dataframe(dtype_df, use_container_width=True)

        # Raw data preview
        st.markdown('<div class="section-header">Raw Data Preview (first 100 rows)</div>', unsafe_allow_html=True)

        preview_cols = [
            "customerID", "gender", "SeniorCitizen", "Partner",
            "tenure", "Contract", "MonthlyCharges", "TotalCharges",
            "InternetService", "Churn", "ChurnFlag",
            "Region", "ProductTier", "TenureBand",
        ]
        st.dataframe(dff[preview_cols].head(100), use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 – Recommendations
    # ══════════════════════════════════════════════════════════════════════════

    with tab_recs:
        st.markdown('<div class="section-header">📋 Business Findings & Recommendations</div>', unsafe_allow_html=True)

        cr_pct = churn_rate(df) * 100
        arpu_v = arpu(df)
        clv_v  = avg_clv(df)
        high_n = len(scored_df[scored_df["RiskTier"] == "High"])
        mtm_cr = churn_by_column(df, "Contract")
        mtm_cr_val = mtm_cr[mtm_cr["Contract"] == "Month-to-month"]["ChurnRate"].values
        mtm_rate = float(mtm_cr_val[0]) if len(mtm_cr_val) else 0.0

        # ── Summary metrics banner
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Overall Churn Rate", f"{cr_pct:.1f}%")
        sc2.metric("Monthly ARPU",       dollar_fmt(arpu_v))
        sc3.metric("Average CLV",        dollar_fmt(clv_v))

        st.markdown("---")

        # ── Findings
        st.markdown("### 🔍 Key Findings")

        findings = [
            (
                "🔴 Month-to-month contracts drive the most churn",
                f"Customers on month-to-month contracts churn at <strong>{mtm_rate:.1f}%</strong>, "
                "compared to far lower rates on annual or two-year plans. Contract flexibility is the single "
                "most powerful churn predictor."
            ),
            (
                "🔴 Fiber optic internet users churn at nearly 2× the rate of DSL customers",
                "Fiber optic customers show high churn despite (or because of) higher bills. "
                "Service quality, pricing perception, or competition may be at play."
            ),
            (
                "🟡 New customers (0–12 months) are extremely vulnerable",
                "The first year is the highest-risk window. Customers who don't onboard well "
                "or don't see value quickly abandon the service before building loyalty."
            ),
            (
                "🟡 Lack of online security & tech support strongly predicts churn",
                "Customers without these add-ons are significantly more likely to leave. "
                "They may feel underserved or unprotected and switch to competitors."
            ),
            (
                "🟡 Electronic check payment method correlates with higher churn",
                "Customers paying via electronic check churn at higher rates — possibly "
                "reflecting a less committed, less automated relationship with the service."
            ),
            (
                "🟢 Long-tenure customers are the most loyal and highest-value",
                f"Customers with 49–72 months tenure have the lowest churn rate and highest CLV "
                f"(avg CLV: {dollar_fmt(clv_v)}). They are the core revenue base."
            ),
        ]

        for title, body in findings:
            st.markdown(
                f'<div class="rec-card"><strong>{title}</strong><br>{body}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### ✅ Recommended Actions")

        recommendations = [
            (
                "1. Launch Contract Upgrade Incentive Programme",
                "Offer month-to-month customers a meaningful discount (15–20%) to switch to annual "
                "contracts. Use the ML churn probability score to prioritise outreach — focus on "
                f"the top {high_n:,} high-risk customers first. Even converting 30% would "
                "materially reduce churn.",
                "🎯 Immediate — High Impact",
            ),
            (
                "2. Create a 90-Day Onboarding Success Programme",
                "New customers in the first 3 months need proactive check-ins, tutorials, "
                "and a dedicated support channel. Set automated triggers at Day 7, 30, and 90 "
                "to ensure adoption and value realisation before the cancellation window opens.",
                "🎯 Short-term — High Impact",
            ),
            (
                "3. Bundle Security & Support Add-ons at Reduced Price",
                "Customers without TechSupport or OnlineSecurity churn significantly more. "
                "Offer these services free or discounted for the first 6 months — they increase "
                "stickiness and customer-perceived value far beyond their cost.",
                "🎯 Short-term — Medium Impact",
            ),
            (
                "4. Investigate Fiber Optic Service Quality",
                "Run NPS surveys specifically targeting fiber optic subscribers who have been "
                "customers for less than 24 months. Address speed, reliability, or pricing "
                "complaints before they escalate to cancellation.",
                "📋 Medium-term — High Impact",
            ),
            (
                "5. Automate Payment & Increase AutoPay Adoption",
                "Electronic check payers churn more. Offer a small monthly discount (e.g. $5) "
                "for switching to automatic credit card or bank transfer payments. This reduces "
                "churn and lowers payment failure rates simultaneously.",
                "🎯 Short-term — Medium Impact",
            ),
            (
                "6. Build a Real-Time Churn Score API",
                "Operationalise the ML model as an internal API that scores every customer "
                "nightly. Feed scores into CRM (e.g. Salesforce) so retention agents see "
                "churn probability on every customer record and can act proactively.",
                "📋 Medium-term — Strategic",
            ),
            (
                "7. Design a Loyalty Rewards Programme for High-Tenure Customers",
                "Customers at 24+ months tenure are extremely valuable. Recognise them with "
                "exclusive benefits (free upgrades, priority support) to sustain loyalty and "
                "generate word-of-mouth referrals that acquire new long-term customers.",
                "📋 Long-term — Strategic",
            ),
        ]

        for title, body, tag in recommendations:
            st.markdown(
                f'<div class="rec-card">'
                f'<strong>{title}</strong> &nbsp;<span style="font-size:0.78rem;color:#64748b;">{tag}</span>'
                f'<br>{body}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Expected impact summary
        st.markdown("### 📊 Expected Business Impact")

        impact_data = {
            "Initiative": [
                "Contract Upgrade Programme",
                "Onboarding Success Programme",
                "Security/Support Bundling",
                "Fiber Optic Service Fix",
                "AutoPay Incentive",
            ],
            "Churn Reduction (est.)": ["8–12%", "5–8%", "3–5%", "4–7%", "2–4%"],
            "Revenue Impact": ["High", "High", "Medium", "High", "Low"],
            "Effort": ["Medium", "Medium", "Low", "High", "Low"],
            "Time to Impact": ["1–3 months", "3–6 months", "1–2 months", "6–12 months", "1 month"],
        }
        st.dataframe(pd.DataFrame(impact_data), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
