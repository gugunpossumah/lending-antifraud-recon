-- ============================================================================
-- 02_fraud_features.sql
-- Customer-Level Fraud Feature Engineering (FPU + FTR)
-- Produces 14 engineered features + target_fraud_label
-- ============================================================================

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
