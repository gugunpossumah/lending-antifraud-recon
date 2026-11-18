import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Lending & Anti-Fraud Reconciliation Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Lending Monitoring, Anti-Fraud (FPU/FTR) & Financial Reconciliation System")
st.caption(
    "End-to-End Analytics Portfolio: SQLite (SQL CTE & 2-Way Reconciliation) + "
    "Out-of-Fold Fraud Risk Scoring + Automated COA Audit"
)


@st.cache_data
def load_data():
    recon = pd.read_csv("summary_reconciliation.csv")
    coa = pd.read_csv("summary_coa_mapping.csv")
    scores = pd.read_csv("summary_customer_fraud_scores.csv")
    importance = pd.read_csv("summary_feature_importance.csv")
    return recon, coa, scores, importance


df_recon, df_coa, df_scores, df_importance = load_data()

# Sidebar Controls (Interactive Decision Thresholds)
st.sidebar.header("Control Panel (Risk Ops)")
high_risk_threshold = st.sidebar.slider(
    "Batas Skor High Risk (Block/Audit)",
    min_value=50,
    max_value=95,
    value=70,
    step=5,
)
medium_risk_threshold = st.sidebar.slider(
    "Batas Skor Medium Risk (Manual Review)",
    min_value=10,
    max_value=high_risk_threshold - 5,
    value=30,
    step=5,
)

df_scores = df_scores.copy()
df_scores["dynamic_risk_tier"] = pd.cut(
    df_scores["fraud_risk_score"],
    bins=[-1, medium_risk_threshold, high_risk_threshold, 100],
    labels=["Low Risk", "Medium Risk (Review)", "High Risk (Block/Audit)"],
)

tab1, tab2, tab3 = st.tabs([
    "1. Anti-Fraud Risk Scoring",
    "2. Lending Reconciliation (2-Way)",
    "3. COA Mapping & Fraud Exposure",
])

# ============================================================================
# TAB 1: ANTI-FRAUD SCORING
# ============================================================================
with tab1:
    st.subheader("Customer Risk Scoring & Top Fraud Drivers")

    high_risk_df = df_scores[df_scores["dynamic_risk_tier"] == "High Risk (Block/Audit)"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Nasabah Dianalisis", f"{len(df_scores):,}")
    col2.metric(
        f"High Risk (Skor > {high_risk_threshold})",
        f"{len(high_risk_df):,}",
        f"{(len(high_risk_df) / max(len(df_scores), 1)) * 100:.1f}% dari total",
    )
    col3.metric("Rata-rata Risk Score", f"{df_scores['fraud_risk_score'].mean():.2f}")
    col4.metric(
        "Historical Fraud Rate",
        f"{df_scores['target_fraud_label'].mean() * 100:.2f}%",
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_hist = px.histogram(
            df_scores,
            x="fraud_risk_score",
            color="dynamic_risk_tier",
            nbins=25,
            title="Distribusi Fraud Risk Score (Out-of-Fold)",
            labels={"fraud_risk_score": "Fraud Risk Score (0-100)", "dynamic_risk_tier": "Risk Tier"},
            color_discrete_map={
                "Low Risk": "#2E7D32",
                "Medium Risk (Review)": "#F9A825",
                "High Risk (Block/Audit)": "#C62828",
            },
        )
        fig_hist.add_vline(x=high_risk_threshold, line_dash="dash", line_color="#C62828")
        st.plotly_chart(fig_hist, use_container_width=True)

    with chart_col2:
        fig_imp = px.bar(
            df_importance.head(10).sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 Faktor Pendorong Risiko Fraud (Feature Importance)",
            labels={"importance": "Importance Score", "feature": "Engineered Feature"},
            color_discrete_sequence=["#1565C0"],
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    tier_filter = st.selectbox(
        "Filter Kategori Risiko:",
        ["Semua", "High Risk (Block/Audit)", "Medium Risk (Review)", "Low Risk"],
    )
    df_view = (
        df_scores
        if tier_filter == "Semua"
        else df_scores[df_scores["dynamic_risk_tier"] == tier_filter]
    )
    df_view = df_view.sort_values("fraud_risk_score", ascending=False)

    st.download_button(
        label="Download Daftar Nasabah Terfilter (CSV)",
        data=df_view.to_csv(index=False).encode("utf-8"),
        file_name="audit_fraud_risk_customers.csv",
        mime="text/csv",
    )
    st.dataframe(df_view, use_container_width=True)

# ============================================================================
# TAB 2: TWO-WAY RECONCILIATION
# ============================================================================
with tab2:
    st.subheader("Rekonsiliasi 2-Arah: Pinjaman Disetujui (FPU) vs Pencairan (FTR)")

    unreconciled_df = df_recon[
        ~df_recon["reconciliation_status"].isin([
            "Reconciled",
            "Reconciled (With Prior Failed Retry)",
        ])
    ]
    unauth_count = (
        df_recon["reconciliation_status"] == "Unauthorized Disbursement (Cair Tanpa Approval)"
    ).sum()

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Approved Amount", f"{df_recon['total_approved_amount'].sum():,.2f}")
    r2.metric("Total Settled Disbursement", f"{df_recon['settled_disb_amount'].sum():,.2f}")
    r3.metric("Total Kasus Unreconciled", f"{len(unreconciled_df):,}")
    r4.metric("Unauthorized Disbursement", f"{unauth_count:,}")

    recon_summary = (
        df_recon.groupby("reconciliation_status", as_index=False)
        .agg(
            customer_count=("customer_id", "count"),
            abs_variance=("unreconciled_variance", lambda s: s.abs().sum()),
        )
        .sort_values("customer_count", ascending=False)
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        fig_recon_cnt = px.bar(
            recon_summary,
            x="customer_count",
            y="reconciliation_status",
            orientation="h",
            title="Jumlah Nasabah per Status Rekonsiliasi",
            labels={"customer_count": "Jumlah Nasabah", "reconciliation_status": "Status"},
            color_discrete_sequence=["#00838F"],
        )
        st.plotly_chart(fig_recon_cnt, use_container_width=True)

    with rc2:
        fig_recon_var = px.bar(
            recon_summary,
            x="abs_variance",
            y="reconciliation_status",
            orientation="h",
            title="Total Nilai Absolut Selisih (Variance) per Status",
            labels={"abs_variance": "Absolut Selisih Nominal", "reconciliation_status": "Status"},
            color_discrete_sequence=["#D84315"],
        )
        st.plotly_chart(fig_recon_var, use_container_width=True)

    status_options = ["Semua"] + sorted(df_recon["reconciliation_status"].unique().tolist())
    selected_status = st.selectbox("Filter Status Rekonsiliasi:", status_options)
    df_recon_view = (
        df_recon
        if selected_status == "Semua"
        else df_recon[df_recon["reconciliation_status"] == selected_status]
    )

    st.download_button(
        label="Download Laporan Rekonsiliasi (CSV)",
        data=df_recon_view.to_csv(index=False).encode("utf-8"),
        file_name="audit_reconciliation_report.csv",
        mime="text/csv",
    )
    st.dataframe(df_recon_view, use_container_width=True)

# ============================================================================
# TAB 3: COA MAPPING & FRAUD EXPOSURE
# ============================================================================
with tab3:
    st.subheader("Pemetaan Otomatis Jurnal Chart of Accounts (COA) & Eksposur Fraud")

    posted_df = df_coa[df_coa["gl_posting_status"] == "Posted to GL"]
    excluded_df = df_coa[df_coa["gl_posting_status"] == "Excluded from GL"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Posted to GL", f"{posted_df['total_nominal_amount'].sum():,.2f}")
    c2.metric("Excluded (Failed Trx)", f"{excluded_df['total_nominal_amount'].sum():,.2f}")
    c3.metric("Total Transaksi Fraud", f"{int(df_coa['flagged_fraud_count'].sum()):,}")
    c4.metric("Total Eksposur Fraud (Nominal)", f"{df_coa['flagged_fraud_amount'].sum():,.2f}")

    gl_filter = st.radio(
        "Filter Status General Ledger (GL):",
        ["Semua", "Posted to GL", "Excluded from GL"],
        horizontal=True,
    )
    df_coa_view = (
        df_coa if gl_filter == "Semua" else df_coa[df_coa["gl_posting_status"] == gl_filter]
    )

    fig_coa = px.bar(
        df_coa_view,
        x="transaction_type",
        y="flagged_fraud_amount",
        color="transaction_status",
        barmode="group",
        title="Eksposur Nominal Fraud per Tipe Transaksi",
        labels={
            "transaction_type": "Tipe Transaksi",
            "flagged_fraud_amount": "Nominal Fraud",
            "transaction_status": "Status Transaksi",
        },
    )
    st.plotly_chart(fig_coa, use_container_width=True)

    st.download_button(
        label="Download Ringkasan Jurnal COA (CSV)",
        data=df_coa_view.to_csv(index=False).encode("utf-8"),
        file_name="summary_coa_audit.csv",
        mime="text/csv",
    )
    st.dataframe(df_coa_view, use_container_width=True)
