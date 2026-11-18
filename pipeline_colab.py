# %% [markdown]
# # Lending Monitoring, Anti-Fraud (FPU/FTR) & Financial Reconciliation Pipeline
# Jalankan script ini di Google Colab (atau lokal) untuk menghasilkan tabel SQLite dan file CSV summary untuk Streamlit Dashboard.

# %%
# 1. Setup & Load Data
import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Jika dijalankan di Google Colab, uncomment 2 baris berikut:
# from google.colab import drive
# drive.mount('/content/drive')

FOLDER_PATH = "/content/drive/MyDrive/2026/Project/Nov/archive (1)/"
DB_PATH = "lending_antifraud_recon.db"

df_loans = pd.read_csv(os.path.join(FOLDER_PATH, "loan_applications.csv"))
df_trx = pd.read_csv(os.path.join(FOLDER_PATH, "transactions.csv"))

print(f"Loan Applications : {df_loans.shape[0]:,} baris | {df_loans.shape[1]} kolom")
print(f"Transactions      : {df_trx.shape[0]:,} baris | {df_trx.shape[1]} kolom")

# %%
# 2. Data Quality Audit (Best Practice untuk Data Analyst)
def audit_dataframe(df: pd.DataFrame, name: str, id_col: str) -> None:
    print(f"\n--- Data Quality Check: {name} ---")
    print(f"Duplicate {id_col}: {df[id_col].duplicated().sum():,}")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        print("Kolom dengan Missing Values:\n", nulls)
    else:
        print("Missing Values: 0 (Bersih)")

audit_dataframe(df_loans, "loan_applications", "application_id")
audit_dataframe(df_trx, "transactions", "transaction_id")

# Hapus duplikat ID jika ada agar agregasi SQL akurat
df_loans = df_loans.drop_duplicates(subset=["application_id"]).copy()
df_trx = df_trx.drop_duplicates(subset=["transaction_id"]).copy()

# %%
# 3. Load ke SQLite & Buat Index agar Query JOIN Cepat
conn = sqlite3.connect(DB_PATH)
df_loans.to_sql("loan_applications", conn, if_exists="replace", index=False)
df_trx.to_sql("transactions", conn, if_exists="replace", index=False)

conn.execute("CREATE INDEX IF NOT EXISTS idx_loans_cust ON loan_applications(customer_id);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_loans_status ON loan_applications(loan_status);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_trx_cust ON transactions(customer_id);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_trx_type_status ON transactions(transaction_type, transaction_status);")
conn.commit()


def run_query(sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


# %%
# 4. SQL Query 1: Two-Way Lending & Disbursement Reconciliation
query_reconciliation = """
WITH approved_loans AS (
    SELECT
        customer_id,
        COUNT(application_id) AS total_approved_apps,
        ROUND(SUM(loan_amount_requested), 2) AS total_approved_amount
    FROM loan_applications
    WHERE loan_status = 'Approved'
    GROUP BY customer_id
),
disbursement_trx AS (
    SELECT
        customer_id,
        COUNT(transaction_id) AS total_disb_trx,
        SUM(CASE WHEN transaction_status = 'Success' THEN 1 ELSE 0 END) AS success_disb_count,
        SUM(CASE WHEN transaction_status = 'Failed' THEN 1 ELSE 0 END) AS failed_disb_count,
        ROUND(SUM(CASE WHEN transaction_status = 'Success' THEN transaction_amount ELSE 0 END), 2) AS settled_disb_amount,
        ROUND(SUM(CASE WHEN transaction_status = 'Failed' THEN transaction_amount ELSE 0 END), 2) AS failed_disb_amount
    FROM transactions
    WHERE transaction_type = 'Loan Disbursement'
    GROUP BY customer_id
),
combined_recon AS (
    SELECT
        a.customer_id,
        a.total_approved_apps,
        a.total_approved_amount,
        COALESCE(d.total_disb_trx, 0) AS total_disb_trx,
        COALESCE(d.success_disb_count, 0) AS success_disb_count,
        COALESCE(d.failed_disb_count, 0) AS failed_disb_count,
        COALESCE(d.settled_disb_amount, 0.0) AS settled_disb_amount,
        COALESCE(d.failed_disb_amount, 0.0) AS failed_disb_amount
    FROM approved_loans a
    LEFT JOIN disbursement_trx d
        ON a.customer_id = d.customer_id

    UNION ALL

    SELECT
        d.customer_id,
        0 AS total_approved_apps,
        0.0 AS total_approved_amount,
        d.total_disb_trx,
        d.success_disb_count,
        d.failed_disb_count,
        d.settled_disb_amount,
        d.failed_disb_amount
    FROM disbursement_trx d
    LEFT JOIN approved_loans a
        ON d.customer_id = a.customer_id
    WHERE a.customer_id IS NULL
)
SELECT
    customer_id,
    total_approved_apps,
    total_approved_amount,
    total_disb_trx,
    success_disb_count,
    failed_disb_count,
    settled_disb_amount,
    failed_disb_amount,
    ROUND(total_approved_amount - settled_disb_amount, 2) AS unreconciled_variance,
    CASE
        WHEN total_approved_apps = 0 AND settled_disb_amount > 0
            THEN 'Unauthorized Disbursement (Cair Tanpa Approval)'
        WHEN total_approved_apps = 0 AND settled_disb_amount = 0
            THEN 'Orphan Failed Disbursement (Gagal Tanpa Approval)'
        WHEN total_disb_trx = 0
            THEN 'Unrecorded Disbursement (Belum Cair di Trx)'
        WHEN ABS(total_approved_amount - settled_disb_amount) <= 0.01 AND failed_disb_count > 0
            THEN 'Reconciled (With Prior Failed Retry)'
        WHEN ABS(total_approved_amount - settled_disb_amount) <= 0.01
            THEN 'Reconciled'
        WHEN settled_disb_amount = 0 AND failed_disb_amount > 0
            THEN 'Disbursement Failed (Butuh Re-trigger)'
        WHEN settled_disb_amount > total_approved_amount + 0.01
            THEN 'Over-Disbursement (Kelebihan Pencairan)'
        ELSE 'Under-Disbursement (Selisih Kurang Cair)'
    END AS reconciliation_status
FROM combined_recon
ORDER BY ABS(total_approved_amount - settled_disb_amount) DESC;
"""

df_recon = run_query(query_reconciliation)
print("\nRingkasan Status Rekonsiliasi:")
print(df_recon["reconciliation_status"].value_counts())

# %%
# 5. SQL Query 2: Customer Fraud Feature Engineering (14 Features)
query_fraud_features = """
WITH fpu_user_features AS (
    SELECT
        customer_id,
        COUNT(application_id) AS f1_total_loan_apps,
        SUM(CASE WHEN loan_status = 'Declined' THEN 1 ELSE 0 END) AS f2_declined_apps_count,
        ROUND(AVG(CASE WHEN loan_status = 'Declined' THEN 1.0 ELSE 0.0 END), 4) AS f3_rejection_rate,
        ROUND(MAX(loan_amount_requested), 2) AS f4_max_loan_requested,
        ROUND(AVG(monthly_income), 2) AS avg_monthly_income,
        MIN(cibil_score) AS f5_min_cibil_score,
        ROUND(MAX(debt_to_income_ratio), 4) AS f6_max_dti_ratio,
        ROUND(
            CAST(MAX(loan_amount_requested) AS REAL) / NULLIF(AVG(monthly_income), 0),
            2
        ) AS f7_loan_to_income_spike,
        MAX(COALESCE(fraud_flag, 0)) AS app_fraud_flag
    FROM loan_applications
    GROUP BY customer_id
),
ftr_trx_features AS (
    SELECT
        customer_id,
        COUNT(transaction_id) AS f8_total_trx_count,
        ROUND(
            CAST(MAX(transaction_amount) AS REAL) / NULLIF(AVG(transaction_amount), 0),
            2
        ) AS f9_trx_amount_spike_ratio,
        SUM(CASE WHEN transaction_status = 'Failed' THEN 1 ELSE 0 END) AS f10_failed_trx_count,
        SUM(COALESCE(is_international_transaction, 0)) AS f11_intl_trx_count,
        COUNT(DISTINCT device_used) AS f12_distinct_devices,
        COUNT(DISTINCT ip_address) AS f13_distinct_ips,
        SUM(CASE WHEN account_balance_after_transaction <= 0 THEN 1 ELSE 0 END) AS f14_zero_balance_drain_count,
        MAX(COALESCE(fraud_flag, 0)) AS trx_fraud_flag
    FROM transactions
    GROUP BY customer_id
)
SELECT
    u.customer_id,
    u.f1_total_loan_apps,
    u.f2_declined_apps_count,
    u.f3_rejection_rate,
    u.f4_max_loan_requested,
    u.avg_monthly_income,
    u.f5_min_cibil_score,
    u.f6_max_dti_ratio,
    COALESCE(u.f7_loan_to_income_spike, 0.0) AS f7_loan_to_income_spike,
    COALESCE(t.f8_total_trx_count, 0) AS f8_total_trx_count,
    COALESCE(t.f9_trx_amount_spike_ratio, 0.0) AS f9_trx_amount_spike_ratio,
    COALESCE(t.f10_failed_trx_count, 0) AS f10_failed_trx_count,
    COALESCE(t.f11_intl_trx_count, 0) AS f11_intl_trx_count,
    COALESCE(t.f12_distinct_devices, 0) AS f12_distinct_devices,
    COALESCE(t.f13_distinct_ips, 0) AS f13_distinct_ips,
    COALESCE(t.f14_zero_balance_drain_count, 0) AS f14_zero_balance_drain_count,
    u.app_fraud_flag,
    COALESCE(t.trx_fraud_flag, 0) AS trx_fraud_flag,
    CASE
        WHEN u.app_fraud_flag = 1 OR COALESCE(t.trx_fraud_flag, 0) = 1 THEN 1
        ELSE 0
    END AS target_fraud_label
FROM fpu_user_features u
LEFT JOIN ftr_trx_features t
    ON u.customer_id = t.customer_id;
"""

df_features = run_query(query_fraud_features)
print(f"\nTotal Customer Features: {len(df_features):,} | Fraud Rate: {df_features['target_fraud_label'].mean() * 100:.2f}%")

# %%
# 6. SQL Query 3: COA Journal Mapping & Financial Exposure
query_coa_mapping = """
SELECT
    transaction_type,
    transaction_status,
    CASE
        WHEN transaction_status = 'Success' THEN 'Posted to GL'
        ELSE 'Excluded from GL'
    END AS gl_posting_status,
    CASE
        WHEN transaction_status != 'Success'
            THEN 'No GL Posting (Failed / Non-Settled Transaction)'
        WHEN transaction_type = 'Loan Disbursement'
            THEN 'Dr 1-1300 Piutang Pinjaman / Cr 1-1100 Kas & Bank'
        WHEN transaction_type = 'Deposit'
            THEN 'Dr 1-1100 Kas & Bank / Cr 2-1100 Simpanan Nasabah'
        WHEN transaction_type IN (
            'Bill Payment', 'UPI', 'Debit Card', 'Credit Card',
            'Net Banking', 'Fund Transfer', 'ATM Withdrawal'
        )
            THEN 'Dr 2-1100 Simpanan Nasabah / Cr 1-1100 Kas & Bank'
        ELSE 'Unmapped Account (Perlu Review Accounting)'
    END AS coa_journal_mapping,
    COUNT(transaction_id) AS total_transactions,
    ROUND(SUM(transaction_amount), 2) AS total_nominal_amount,
    SUM(CASE WHEN fraud_flag = 1 THEN 1 ELSE 0 END) AS flagged_fraud_count,
    ROUND(
        SUM(CASE WHEN fraud_flag = 1 THEN transaction_amount ELSE 0 END),
        2
    ) AS flagged_fraud_amount
FROM transactions
GROUP BY transaction_type, transaction_status
ORDER BY gl_posting_status DESC, total_nominal_amount DESC;
"""

df_coa_summary = run_query(query_coa_mapping)
print("\nRingkasan COA Mapping:")
print(df_coa_summary.head(10))

# %%
# 7. Fraud Risk Scoring (Out-of-Fold Evaluation + Target Leakage Check)
all_14_features = [
    "f1_total_loan_apps", "f2_declined_apps_count", "f3_rejection_rate",
    "f4_max_loan_requested", "f5_min_cibil_score", "f6_max_dti_ratio",
    "f7_loan_to_income_spike", "f8_total_trx_count", "f9_trx_amount_spike_ratio",
    "f10_failed_trx_count", "f11_intl_trx_count", "f12_distinct_devices",
    "f13_distinct_ips", "f14_zero_balance_drain_count",
]

# Fitur pre-decision (mengeluarkan f2 & f3 untuk cek potensi target leakage dari status Declined)
pre_decision_12_features = [
    f for f in all_14_features if f not in ("f2_declined_apps_count", "f3_rejection_rate")
]

y = df_features["target_fraud_label"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_feature_set(feature_list: list, label: str):
    X_sub = df_features[feature_list].fillna(0)
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    oof_probs = cross_val_predict(model, X_sub, y, cv=cv, method="predict_proba")[:, 1]
    oof_preds = (oof_probs >= 0.5).astype(int)

    roc = roc_auc_score(y, oof_probs)
    pr = average_precision_score(y, oof_probs)
    print(f"\n=== Evaluasi 5-Fold Out-of-Fold: {label} ===")
    print(f"ROC-AUC : {roc:.4f} | PR-AUC (Average Precision): {pr:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y, oof_preds))
    print(classification_report(y, oof_preds, digits=4))

    model.fit(X_sub, y)
    return model, oof_probs

# Bandingkan keduanya agar bisa dijelaskan saat interview Data Analyst!
model_14, oof_probs_14 = evaluate_feature_set(all_14_features, "All 14 Features")
model_12, oof_probs_12 = evaluate_feature_set(pre_decision_12_features, "Pre-Decision 12 Features (Anti-Leakage)")

# Gunakan model All 14 Features untuk scoring final (Post-Application / Ongoing Monitoring)
# Benchmark 5-Fold OOF: All 14 Features (ROC-AUC 0.9135 | PR-AUC 0.7131 | F1 0.7019 | TP 246 / FN 120)
# mengungguli Pre-Decision 12 Features (ROC-AUC 0.7673 | PR-AUC 0.5203 | F1 0.6346 | TP 178 / FN 188)
SELECTED_FEATURES = all_14_features
SELECTED_MODEL = model_14
SELECTED_OOF_PROBS = oof_probs_14

df_features["fraud_risk_score"] = np.round(SELECTED_OOF_PROBS * 100, 2)
df_features["risk_tier"] = pd.cut(
    df_features["fraud_risk_score"],
    bins=[-1, 30, 70, 100],
    labels=["Low Risk", "Medium Risk (Review)", "High Risk (Block/Audit)"],
)

# Ekstrak Feature Importance untuk Explainability di Dashboard
df_importance = pd.DataFrame({
    "feature": SELECTED_FEATURES,
    "importance": np.round(SELECTED_MODEL.feature_importances_, 4),
}).sort_values("importance", ascending=False)

# Simpan ke SQLite & Ekspor CSV ringan untuk GitHub / Streamlit Cloud
df_features.to_sql("customer_fraud_scores", conn, if_exists="replace", index=False)
df_recon.to_sql("reconciliation_results", conn, if_exists="replace", index=False)
df_coa_summary.to_sql("coa_mapping_summary", conn, if_exists="replace", index=False)
df_importance.to_sql("feature_importance", conn, if_exists="replace", index=False)
conn.close()

df_recon.to_csv("summary_reconciliation.csv", index=False)
df_coa_summary.to_csv("summary_coa_mapping.csv", index=False)
df_features.to_csv("summary_customer_fraud_scores.csv", index=False)
df_importance.to_csv("summary_feature_importance.csv", index=False)

print("\nSelesai! File summary CSV berhasil diekspor dan siap dipakai Streamlit.")
