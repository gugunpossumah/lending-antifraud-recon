import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Lending & Anti-Fraud Reconciliation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# HELPERS & DATA LOADING
# ============================================================================
FEATURE_LABELS = {
    "f1_total_loan_apps": "Total Loan Applications (f1)",
    "f2_declined_apps_count": "Declined Applications Count (f2)",
    "f3_rejection_rate": "Application Rejection Rate (f3)",
    "f4_max_loan_requested": "Max Loan Requested (f4)",
    "f5_min_cibil_score": "Min CIBIL Credit Score (f5)",
    "f6_max_dti_ratio": "Max Debt-to-Income Ratio (f6)",
    "f7_loan_to_income_spike": "Loan-to-Income Spike (f7)",
    "f8_total_trx_count": "Total Transactions Count (f8)",
    "f9_trx_amount_spike_ratio": "Trx Amount Spike Ratio (f9)",
    "f10_failed_trx_count": "Failed Transactions Count (f10)",
    "f11_intl_trx_count": "International Trx Count (f11)",
    "f12_distinct_devices": "Distinct Devices Used (f12)",
    "f13_distinct_ips": "Distinct IP Addresses (f13)",
    "f14_zero_balance_drain_count": "Zero-Balance Drain Count (f14)",
}


def format_compact(val: float) -> str:
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1_000_000_000:
        return f"{sign}{abs_val / 1_000_000_000:.2f}B"
    if abs_val >= 1_000_000:
        return f"{sign}{abs_val / 1_000_000:.2f}M"
    if abs_val >= 1_000:
        return f"{sign}{abs_val / 1_000:.1f}K"
    return f"{val:,.2f}"


def style_plotly(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=15, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@st.cache_data
def load_data():
    recon = pd.read_csv("summary_reconciliation.csv")
    coa = pd.read_csv("summary_coa_mapping.csv")
    scores = pd.read_csv("summary_customer_fraud_scores.csv")
    importance = pd.read_csv("summary_feature_importance.csv")
    return recon, coa, scores, importance


df_recon, df_coa, df_scores, df_importance = load_data()

# ============================================================================
# HEADER
# ============================================================================
st.title("Lending Monitoring, Anti-Fraud & Financial Reconciliation")
st.caption(
    "FPU/FTR Risk Scoring, Two-Way Disbursement Reconciliation, dan Audit Jurnal COA"
)

# ============================================================================
# SIDEBAR: CONTROL PANEL & MODEL SUMMARY
# ============================================================================
st.sidebar.header("Risk Threshold Settings")
st.sidebar.caption("Sesuaikan batas skor untuk simulasi beban review dan tingkat deteksi fraud.")

high_risk_threshold = st.sidebar.slider(
    "Batas High Risk (Block / Audit)",
    min_value=40,
    max_value=95,
    value=70,
    step=5,
)
medium_risk_threshold = st.sidebar.slider(
    "Batas Medium Risk (Manual Review)",
    min_value=10,
    max_value=high_risk_threshold - 5,
    value=min(30, high_risk_threshold - 5),
    step=5,
)

df_scores = df_scores.copy()
df_scores["dynamic_risk_tier"] = pd.cut(
    df_scores["fraud_risk_score"],
    bins=[-1, medium_risk_threshold, high_risk_threshold, 100],
    labels=["Low Risk", "Medium Risk (Review)", "High Risk (Block/Audit)"],
)

high_risk_df = df_scores[df_scores["dynamic_risk_tier"] == "High Risk (Block/Audit)"]
total_actual_frauds = int(df_scores["target_fraud_label"].sum())
captured_frauds_high = int(high_risk_df["target_fraud_label"].sum())
capture_rate_high = (captured_frauds_high / total_actual_frauds * 100) if total_actual_frauds else 0.0
precision_high = (captured_frauds_high / len(high_risk_df) * 100) if len(high_risk_df) else 0.0

st.sidebar.divider()
st.sidebar.subheader("Model Summary (5-Fold OOF)")
st.sidebar.markdown(
    """
    - **Model:** Random Forest (14 Features)
    - **ROC-AUC:** `0.9135` *(baseline 12 fitur: 0.7673)*
    - **PR-AUC:** `0.7131` *(baseline 12 fitur: 0.5203)*
    - **F1-Score (Fraud):** `0.7019`
    - **Data:** `18,314` nasabah | `50,000` transaksi
    """
)
st.sidebar.caption(
    f"Pada threshold >{high_risk_threshold}, {len(high_risk_df):,} nasabah "
    f"({len(high_risk_df) / len(df_scores) * 100:.2f}%) masuk antrean High Risk "
    f"dengan recall {capture_rate_high:.1f}% dan precision {precision_high:.1f}%."
)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Anti-Fraud Scoring",
    "2. Disbursement Reconciliation",
    "3. COA & Fraud Exposure",
    "4. Customer Audit Lookup",
])

# ============================================================================
# TAB 1: ANTI-FRAUD RISK SCORING
# ============================================================================
with tab1:
    st.subheader("Customer Risk Scoring & Threshold Simulation")

    top4_share = df_importance.head(4)["importance"].sum() * 100
    st.info(
        f"Ringkasan: Dari {len(df_scores):,} nasabah (fraud rate "
        f"{df_scores['target_fraud_label'].mean() * 100:.2f}% / {total_actual_frauds:,} kasus), "
        f"4 fitur teratas menyumbang {top4_share:.1f}% bobot model, terutama "
        f"Failed Transactions Count (35.3%) dan Min CIBIL Score (17.2%)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total Nasabah",
        f"{len(df_scores):,}",
        f"Rata-rata skor: {df_scores['fraud_risk_score'].mean():.2f}",
        delta_color="off",
    )
    c2.metric(
        f"High Risk (Skor > {high_risk_threshold})",
        f"{len(high_risk_df):,}",
        f"{len(high_risk_df) / max(len(df_scores), 1) * 100:.2f}% dari total",
        delta_color="off",
    )
    c3.metric(
        "Fraud Capture Rate (Recall)",
        f"{capture_rate_high:.1f}%",
        f"{captured_frauds_high:,} dari {total_actual_frauds:,} kasus",
        delta_color="off",
        help="Proporsi kasus fraud aktual yang berhasil masuk kategori High Risk pada threshold saat ini.",
    )
    c4.metric(
        "Alert Precision",
        f"{precision_high:.1f}%",
        f"{len(high_risk_df) - captured_frauds_high:,} false alarm",
        delta_color="off",
        help="Proporsi nasabah High Risk yang benar-benar berlabel fraud.",
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_hist = px.histogram(
            df_scores,
            x="fraud_risk_score",
            color="dynamic_risk_tier",
            nbins=25,
            title="Distribusi Out-of-Fold Fraud Risk Score (0-100)",
            labels={
                "fraud_risk_score": "Fraud Risk Score",
                "dynamic_risk_tier": "Risk Tier",
                "count": "Jumlah Nasabah",
            },
            color_discrete_map={
                "Low Risk": "#2E7D32",
                "Medium Risk (Review)": "#F9A825",
                "High Risk (Block/Audit)": "#C62828",
            },
        )
        fig_hist.add_vline(
            x=medium_risk_threshold,
            line_dash="dot",
            line_color="#F9A825",
            annotation_text=f"Med ({medium_risk_threshold})",
            annotation_position="top",
        )
        fig_hist.add_vline(
            x=high_risk_threshold,
            line_dash="dash",
            line_color="#C62828",
            annotation_text=f"High ({high_risk_threshold})",
            annotation_position="top",
        )
        st.plotly_chart(style_plotly(fig_hist), use_container_width=True)

    with chart_col2:
        df_imp_plot = df_importance.head(10).copy()
        df_imp_plot["feature_label"] = df_imp_plot["feature"].map(
            lambda x: FEATURE_LABELS.get(x, x)
        )
        df_imp_plot["importance_pct"] = df_imp_plot["importance"] * 100
        df_imp_plot = df_imp_plot.sort_values("importance_pct", ascending=True)

        fig_imp = px.bar(
            df_imp_plot,
            x="importance_pct",
            y="feature_label",
            orientation="h",
            text=df_imp_plot["importance_pct"].map(lambda v: f"{v:.1f}%"),
            title="Top 10 Faktor Pendorong Risiko Fraud",
            labels={"importance_pct": "Kontribusi Model (%)", "feature_label": ""},
            color="importance_pct",
            color_continuous_scale="Blues",
        )
        fig_imp.update_traces(textposition="outside")
        fig_imp.update_layout(
            coloraxis_showscale=False,
            xaxis_range=[0, df_imp_plot["importance_pct"].max() * 1.18],
        )
        st.plotly_chart(style_plotly(fig_imp), use_container_width=True)

    st.markdown("#### Daftar Nasabah untuk Review")
    f_col1, f_col2 = st.columns([2, 2])
    with f_col1:
        tier_filter = st.selectbox(
            "Filter Kategori Risiko:",
            ["Semua", "High Risk (Block/Audit)", "Medium Risk (Review)", "Low Risk"],
        )
    with f_col2:
        cust_search = st.text_input("Cari Customer ID:", placeholder="Contoh: CUST100001")

    df_view = (
        df_scores
        if tier_filter == "Semua"
        else df_scores[df_scores["dynamic_risk_tier"] == tier_filter]
    )
    if cust_search.strip():
        df_view = df_view[
            df_view["customer_id"].astype(str).str.contains(cust_search.strip(), case=False, na=False)
        ]

    df_view = df_view.sort_values("fraud_risk_score", ascending=False)
    display_cols = [
        "customer_id",
        "fraud_risk_score",
        "dynamic_risk_tier",
        "target_fraud_label",
        "f10_failed_trx_count",
        "f5_min_cibil_score",
        "f2_declined_apps_count",
        "f3_rejection_rate",
        "f4_max_loan_requested",
        "f6_max_dti_ratio",
        "f8_total_trx_count",
    ]

    st.download_button(
        label=f"Export CSV ({len(df_view):,} baris)",
        data=df_view.to_csv(index=False).encode("utf-8"),
        file_name="audit_fraud_risk_customers.csv",
        mime="text/csv",
    )
    st.dataframe(
        df_view[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "customer_id": st.column_config.TextColumn("Customer ID"),
            "fraud_risk_score": st.column_config.ProgressColumn(
                "Fraud Risk Score",
                min_value=0,
                max_value=100,
                format="%.2f",
            ),
            "dynamic_risk_tier": st.column_config.TextColumn("Risk Tier"),
            "target_fraud_label": st.column_config.NumberColumn("Actual Fraud", format="%d"),
            "f10_failed_trx_count": st.column_config.NumberColumn("Failed Trx", format="%d"),
            "f5_min_cibil_score": st.column_config.NumberColumn("Min CIBIL", format="%d"),
            "f2_declined_apps_count": st.column_config.NumberColumn("Declined Apps", format="%d"),
            "f3_rejection_rate": st.column_config.NumberColumn("Rejection Rate", format="%.2f"),
            "f4_max_loan_requested": st.column_config.NumberColumn("Max Loan Req", format="%,.2f"),
            "f6_max_dti_ratio": st.column_config.NumberColumn("Max DTI", format="%.2f"),
            "f8_total_trx_count": st.column_config.NumberColumn("Total Trx", format="%d"),
        },
    )

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
    unauth_df = df_recon[
        df_recon["reconciliation_status"] == "Unauthorized Disbursement (Cair Tanpa Approval)"
    ]
    orphan_failed_count = (
        df_recon["reconciliation_status"] == "Orphan Failed Disbursement (Gagal Tanpa Approval)"
    ).sum()

    unauth_nominal = unauth_df["settled_disb_amount"].sum()
    total_approved = df_recon["total_approved_amount"].sum()
    total_settled = df_recon["settled_disb_amount"].sum()

    if len(unauth_df) > 0:
        st.error(
            f"Temuan Audit: Rekonsiliasi 2-arah mendeteksi {len(unauth_df):,} kasus "
            f"Unauthorized Disbursement (pencairan Success tanpa approval) senilai "
            f"{unauth_nominal:,.2f}, serta {orphan_failed_count:,} kasus Orphan Failed Disbursement."
        )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Total Approved Amount",
        format_compact(total_approved),
        help=f"Nominal persis: {total_approved:,.2f}",
    )
    r2.metric(
        "Total Settled Disbursement",
        format_compact(total_settled),
        help=f"Nominal persis: {total_settled:,.2f}",
    )
    r3.metric(
        "Kasus Unreconciled",
        f"{len(unreconciled_df):,}",
        f"dari {len(df_recon):,} nasabah",
        delta_color="off",
    )
    r4.metric(
        "Unauthorized Disbursement",
        f"{len(unauth_df):,} kasus",
        f"Nominal: {format_compact(unauth_nominal)}",
        delta_color="off",
        help=f"Total pencairan tanpa approval: {unauth_nominal:,.2f}",
    )

    use_log_scale = st.toggle(
        "Gunakan skala logaritmik pada grafik variance",
        value=True,
        help="Memudahkan perbandingan antara selisih miliaran (Unrecorded) dan jutaan (Unauthorized).",
    )

    recon_summary = (
        df_recon.groupby("reconciliation_status", as_index=False)
        .agg(
            customer_count=("customer_id", "count"),
            abs_variance=("unreconciled_variance", lambda s: s.abs().sum()),
        )
        .sort_values("customer_count", ascending=True)
    )
    recon_summary["priority_group"] = recon_summary["reconciliation_status"].apply(
        lambda s: "Prioritas Tinggi (Unauthorized)" if "Unauthorized" in s else "Variance Lainnya"
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        fig_recon_cnt = px.bar(
            recon_summary,
            x="customer_count",
            y="reconciliation_status",
            color="priority_group",
            orientation="h",
            text=recon_summary["customer_count"].map(lambda v: f"{v:,}"),
            title="Jumlah Nasabah per Status Rekonsiliasi",
            labels={
                "customer_count": "Jumlah Nasabah",
                "reconciliation_status": "",
                "priority_group": "Kategori",
            },
            color_discrete_map={
                "Prioritas Tinggi (Unauthorized)": "#C62828",
                "Variance Lainnya": "#00838F",
            },
        )
        fig_recon_cnt.update_traces(textposition="outside")
        fig_recon_cnt.update_layout(xaxis_range=[0, recon_summary["customer_count"].max() * 1.18])
        st.plotly_chart(style_plotly(fig_recon_cnt), use_container_width=True)

    with rc2:
        recon_var_plot = recon_summary[recon_summary["abs_variance"] > 0].sort_values(
            "abs_variance", ascending=True
        )
        fig_recon_var = px.bar(
            recon_var_plot,
            x="abs_variance",
            y="reconciliation_status",
            color="priority_group",
            orientation="h",
            log_x=use_log_scale,
            text=recon_var_plot["abs_variance"].map(format_compact),
            title="Total Nilai Absolut Selisih (Variance)",
            labels={
                "abs_variance": "Absolut Selisih Nominal" + (" (Log Scale)" if use_log_scale else ""),
                "reconciliation_status": "",
                "priority_group": "Kategori",
            },
            color_discrete_map={
                "Prioritas Tinggi (Unauthorized)": "#C62828",
                "Variance Lainnya": "#EF6C00",
            },
        )
        fig_recon_var.update_traces(textposition="outside")
        st.plotly_chart(style_plotly(fig_recon_var), use_container_width=True)

    status_options = ["Semua"] + sorted(df_recon["reconciliation_status"].unique().tolist())
    selected_status = st.selectbox("Filter Status Rekonsiliasi:", status_options)
    df_recon_view = (
        df_recon
        if selected_status == "Semua"
        else df_recon[df_recon["reconciliation_status"] == selected_status]
    )

    st.download_button(
        label=f"Export Laporan Rekonsiliasi ({len(df_recon_view):,} baris)",
        data=df_recon_view.to_csv(index=False).encode("utf-8"),
        file_name="audit_reconciliation_report.csv",
        mime="text/csv",
    )
    st.dataframe(
        df_recon_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "customer_id": st.column_config.TextColumn("Customer ID"),
            "total_approved_apps": st.column_config.NumberColumn("Approved Apps", format="%d"),
            "total_approved_amount": st.column_config.NumberColumn("Approved Amount", format="%,.2f"),
            "total_disb_trx": st.column_config.NumberColumn("Disb Trx", format="%d"),
            "success_disb_count": st.column_config.NumberColumn("Success Disb", format="%d"),
            "failed_disb_count": st.column_config.NumberColumn("Failed Disb", format="%d"),
            "settled_disb_amount": st.column_config.NumberColumn("Settled Disb", format="%,.2f"),
            "failed_disb_amount": st.column_config.NumberColumn("Failed Disb Amt", format="%,.2f"),
            "unreconciled_variance": st.column_config.NumberColumn("Variance", format="%,.2f"),
            "reconciliation_status": st.column_config.TextColumn("Reconciliation Status"),
        },
    )

# ============================================================================
# TAB 3: COA JOURNAL MAPPING & FRAUD FINANCIAL EXPOSURE
# ============================================================================
with tab3:
    st.subheader("Pemetaan Jurnal Chart of Accounts (COA) & Eksposur Fraud")

    posted_df = df_coa[df_coa["gl_posting_status"] == "Posted to GL"]
    excluded_df = df_coa[df_coa["gl_posting_status"] == "Excluded from GL"]
    unmapped_df = df_coa[df_coa["coa_journal_mapping"].str.contains("Unmapped", case=False, na=False)]

    posted_amt = posted_df["total_nominal_amount"].sum()
    excluded_amt = excluded_df["total_nominal_amount"].sum()
    excluded_trx = int(excluded_df["total_transactions"].sum())
    excluded_fraud_trx = int(excluded_df["flagged_fraud_count"].sum())
    realized_fraud_amt = posted_df["flagged_fraud_amount"].sum()
    realized_fraud_cnt = int(posted_df["flagged_fraud_count"].sum())
    total_fraud_amt = df_coa["flagged_fraud_amount"].sum()
    total_fraud_cnt = int(df_coa["flagged_fraud_count"].sum())

    if len(unmapped_df) > 0:
        unmapped_types = ", ".join(unmapped_df["transaction_type"].unique())
        unmapped_trx = int(unmapped_df["total_transactions"].sum())
        unmapped_amt = unmapped_df["total_nominal_amount"].sum()
        st.warning(
            f"Catatan Audit GL: Terdapat {unmapped_trx:,} transaksi {unmapped_types} senilai "
            f"{unmapped_amt:,.2f} yang berstatus Unmapped Account "
            f"(usulan jurnal: Dr 1-1100 Kas & Bank / Cr 1-1300 Piutang Pinjaman). "
            f"Seluruh {excluded_fraud_trx:,} transaksi Failed yang dikecualikan dari GL terindikasi fraud (100%)."
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Posted to GL (Success)",
        format_compact(posted_amt),
        f"{int(posted_df['total_transactions'].sum()):,} transaksi",
        delta_color="off",
        help=f"Nominal persis: {posted_amt:,.2f}",
    )
    k2.metric(
        "Excluded from GL (Failed)",
        format_compact(excluded_amt),
        f"{excluded_trx:,} trx (100% Fraud)",
        delta_color="off",
        help=f"Nominal persis: {excluded_amt:,.2f}",
    )
    k3.metric(
        "Fraud Lolos ke GL",
        format_compact(realized_fraud_amt),
        f"{realized_fraud_cnt:,} transaksi Success",
        delta_color="off",
        help=f"Transaksi fraud yang berstatus Success dan masuk GL: {realized_fraud_amt:,.2f}",
    )
    k4.metric(
        "Total Eksposur Fraud",
        format_compact(total_fraud_amt),
        f"{total_fraud_cnt:,} transaksi",
        delta_color="off",
        help=f"Gabungan fraud Success + Failed: {total_fraud_amt:,.2f}",
    )

    gl_filter = st.radio(
        "Filter Status General Ledger (GL):",
        ["Semua", "Posted to GL", "Excluded from GL"],
        horizontal=True,
    )
    df_coa_view = (
        df_coa if gl_filter == "Semua" else df_coa[df_coa["gl_posting_status"] == gl_filter]
    )

    coa_col1, coa_col2 = st.columns([3, 2])
    with coa_col1:
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
                "transaction_status": "Status",
            },
            color_discrete_map={"Success": "#D84315", "Failed": "#546E7A"},
        )
        fig_coa.update_xaxes(tickangle=-25)
        st.plotly_chart(style_plotly(fig_coa), use_container_width=True)

    with coa_col2:
        fraud_split = pd.DataFrame({
            "Kategori": [
                "Dicegah / Excluded (Failed)",
                "Lolos ke GL (Success)",
            ],
            "Nominal": [excluded_df["flagged_fraud_amount"].sum(), realized_fraud_amt],
        })
        fig_donut = px.pie(
            fraud_split,
            names="Kategori",
            values="Nominal",
            hole=0.52,
            title="Proporsi Eksposur Fraud: Dicegah vs Lolos GL",
            color="Kategori",
            color_discrete_map={
                "Dicegah / Excluded (Failed)": "#2E7D32",
                "Lolos ke GL (Success)": "#C62828",
            },
        )
        fig_donut.update_traces(textinfo="percent+label")
        st.plotly_chart(style_plotly(fig_donut), use_container_width=True)

    st.download_button(
        label="Export Ringkasan COA (CSV)",
        data=df_coa_view.to_csv(index=False).encode("utf-8"),
        file_name="summary_coa_audit.csv",
        mime="text/csv",
    )
    st.dataframe(
        df_coa_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "transaction_type": st.column_config.TextColumn("Transaction Type"),
            "transaction_status": st.column_config.TextColumn("Status"),
            "gl_posting_status": st.column_config.TextColumn("GL Status"),
            "coa_journal_mapping": st.column_config.TextColumn("COA Journal Mapping"),
            "total_transactions": st.column_config.NumberColumn("Total Trx", format="%d"),
            "total_nominal_amount": st.column_config.NumberColumn("Total Nominal", format="%,.2f"),
            "flagged_fraud_count": st.column_config.NumberColumn("Fraud Count", format="%d"),
            "flagged_fraud_amount": st.column_config.NumberColumn("Fraud Exposure", format="%,.2f"),
        },
    )

# ============================================================================
# TAB 4: CUSTOMER AUDIT LOOKUP
# ============================================================================
with tab4:
    st.subheader("Customer Audit Lookup (Profil Risiko & Rekonsiliasi)")
    st.caption(
        "Cek detail satu Customer ID untuk melihat skor risiko, indikator kredit/transaksi, "
        "dan status pencairannya secara bersamaan."
    )

    preset_mode = st.radio(
        "Sumber Pencarian:",
        [
            "Top 50 High-Risk Customers",
            "Top 50 Unauthorized Disbursement",
            "Input Manual Customer ID",
        ],
        horizontal=True,
    )

    selected_cust = None
    if preset_mode == "Top 50 High-Risk Customers":
        top_hr_ids = (
            df_scores.sort_values("fraud_risk_score", ascending=False)["customer_id"]
            .head(50)
            .tolist()
        )
        selected_cust = st.selectbox("Pilih Nasabah High-Risk:", top_hr_ids)
    elif preset_mode == "Top 50 Unauthorized Disbursement":
        top_unauth_ids = (
            df_recon[
                df_recon["reconciliation_status"]
                == "Unauthorized Disbursement (Cair Tanpa Approval)"
            ]
            .sort_values("settled_disb_amount", ascending=False)["customer_id"]
            .head(50)
            .tolist()
        )
        selected_cust = st.selectbox("Pilih Nasabah Unauthorized Disbursement:", top_unauth_ids)
    else:
        all_cust_ids = sorted(
            set(df_scores["customer_id"].astype(str)).union(set(df_recon["customer_id"].astype(str)))
        )
        manual_input = st.text_input(
            "Masukkan Customer ID:",
            value=str(all_cust_ids[0]) if all_cust_ids else "",
        ).strip()
        if manual_input:
            selected_cust = manual_input

    if selected_cust:
        score_match = df_scores[df_scores["customer_id"].astype(str) == str(selected_cust)]
        recon_match = df_recon[df_recon["customer_id"].astype(str) == str(selected_cust)]

        if score_match.empty and recon_match.empty:
            st.warning(f"Customer ID {selected_cust} tidak ditemukan di data Fraud maupun Rekonsiliasi.")
        else:
            left_col, right_col = st.columns([1.1, 1.4])

            with left_col:
                st.markdown(f"#### Profil Risiko: `{selected_cust}`")
                if not score_match.empty:
                    row_s = score_match.iloc[0]
                    score_val = float(row_s["fraud_risk_score"])
                    tier_val = str(row_s["dynamic_risk_tier"])

                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=score_val,
                            title={"text": f"Fraud Risk Score ({tier_val})"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#1E293B"},
                                "steps": [
                                    {"range": [0, medium_risk_threshold], "color": "#C8E6C9"},
                                    {"range": [medium_risk_threshold, high_risk_threshold], "color": "#FFF9C4"},
                                    {"range": [high_risk_threshold, 100], "color": "#FFCDD2"},
                                ],
                                "threshold": {
                                    "line": {"color": "#C62828", "width": 4},
                                    "thickness": 0.75,
                                    "value": high_risk_threshold,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(height=270, margin=dict(l=20, r=20, t=50, b=10))
                    st.plotly_chart(style_plotly(fig_gauge), use_container_width=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Failed Trx", f"{int(row_s['f10_failed_trx_count'])}")
                    m2.metric("Min CIBIL", f"{int(row_s['f5_min_cibil_score'])}")
                    m3.metric("Rejection Rate", f"{float(row_s['f3_rejection_rate']) * 100:.0f}%")

                    m4, m5, m6 = st.columns(3)
                    m4.metric("Max DTI Ratio", f"{float(row_s['f6_max_dti_ratio']):.2f}")
                    m5.metric(
                        "Devices / IPs",
                        f"{int(row_s['f12_distinct_devices'])} / {int(row_s['f13_distinct_ips'])}",
                    )
                    m6.metric(
                        "Label Fraud",
                        "Ya (1)" if int(row_s["target_fraud_label"]) == 1 else "Tidak (0)",
                    )
                else:
                    st.error(
                        "Nasabah ini tidak memiliki riwayat pengajuan pinjaman (FPU), "
                        "namun tercatat menerima transaksi pencairan dana."
                    )

            with right_col:
                st.markdown("#### Status Rekonsiliasi & Rekomendasi")
                if not recon_match.empty:
                    row_r = recon_match.iloc[0]
                    status_r = str(row_r["reconciliation_status"])

                    if "Unauthorized" in status_r:
                        st.error(f"Status Rekonsiliasi: {status_r}")
                    elif "Reconciled" in status_r:
                        st.success(f"Status Rekonsiliasi: {status_r}")
                    else:
                        st.warning(f"Status Rekonsiliasi: {status_r}")

                    rc_a, rc_b, rc_c = st.columns(3)
                    rc_a.metric(
                        "Approved Amount",
                        format_compact(float(row_r["total_approved_amount"])),
                        help=f"{float(row_r['total_approved_amount']):,.2f}",
                    )
                    rc_b.metric(
                        "Settled Disb",
                        format_compact(float(row_r["settled_disb_amount"])),
                        help=f"{float(row_r['settled_disb_amount']):,.2f}",
                    )
                    rc_c.metric(
                        "Variance",
                        format_compact(float(row_r["unreconciled_variance"])),
                        help=f"{float(row_r['unreconciled_variance']):,.2f}",
                    )
                else:
                    st.info("Tidak ada pinjaman Approved maupun transaksi Loan Disbursement.")

                actions = []
                if not recon_match.empty and "Unauthorized" in str(recon_match.iloc[0]["reconciliation_status"]):
                    actions.append(
                        "**Eskalasi Finance & Audit:** Verifikasi sumber pencairan dana karena tidak ditemukan record approval."
                    )
                if not score_match.empty:
                    s_val = float(score_match.iloc[0]["fraud_risk_score"])
                    if s_val > high_risk_threshold:
                        actions.append(
                            "**Block / Hold Akun:** Skor risiko melewati batas High Risk, perlu investigasi sebelum pencairan/transaksi lanjutan."
                        )
                    elif s_val > medium_risk_threshold:
                        actions.append(
                            "**Manual Review:** Lakukan pengecekan KYC, riwayat pengajuan, dan perangkat yang digunakan."
                        )
                    if int(score_match.iloc[0]["f10_failed_trx_count"]) > 0:
                        actions.append(
                            "**Audit Failed Transactions:** Terdapat riwayat transaksi gagal yang berkorelasi kuat dengan indikasi fraud."
                        )

                if not actions:
                    actions.append(
                        "**Monitor Normal:** Profil nasabah berada di batas aman, belum perlu eskalasi."
                    )

                st.markdown("**Tindak Lanjut yang Disarankan:**")
                for act in actions:
                    st.markdown(f"- {act}")
