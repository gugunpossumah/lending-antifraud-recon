-- ============================================================================
-- 01_reconciliation.sql
-- Two-Way Lending & Disbursement Reconciliation (FPU vs FTR)
-- Compatible with SQLite 3.35+ (uses LEFT JOIN + UNION ALL for FULL OUTER JOIN)
-- ============================================================================

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
    -- Direction 1: Approved loans matched against disbursements
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

    -- Direction 2: Orphan / Unauthorized disbursements without any Approved loan
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
