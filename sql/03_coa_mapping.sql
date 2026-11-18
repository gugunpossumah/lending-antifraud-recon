-- ============================================================================
-- 03_coa_mapping.sql
-- Automated Chart of Accounts (COA) Mapping & Fraud Financial Exposure
-- Ensures Failed/Non-Settled transactions are excluded from General Ledger (GL)
-- ============================================================================

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
