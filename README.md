# Lending Monitoring, Anti-Fraud (FPU/FTR) & Financial Reconciliation System

End-to-end **Data Analytics & Financial Control** project yang mengintegrasikan **SQL (SQLite)**, **Python (`pandas`, `scikit-learn`)**, dan **Interactive BI Dashboard (`Streamlit` + `Plotly`)** untuk mendeteksi anomali pencairan dana, risiko fraud nasabah, dan memastikan akurasi jurnal akuntansi (COA).

---

## Business Problem
Perusahaan fintech lending menghadapi 3 tantangan operasional utama:
1. **Disbursement Leakage & Reconciliation Gaps:** Ketidaksesuaian antara pengajuan pinjaman yang disetujui (*FPU / Loan Applications*) dengan realisasi pencairan dana (*FTR / Transactions*), termasuk risiko pencairan tanpa persetujuan (*Unauthorized Disbursement*) atau pencairan gagal yang belum di-retry.
2. **Multi-Channel Fraud Detection:** Sulitnya memprioritaskan nasabah berisiko tinggi karena sinyal fraud tersebar di data pengajuan kredit (rasio utang, skor kredit) dan perilaku transaksi (lonjakan nominal, banyak perangkat/IP, transaksi internasional).
3. **General Ledger (GL) Audit Accuracy:** Transaksi berstatus `Failed` berisiko ikut tercatat ke jurnal akuntansi jika pemetaan Chart of Accounts (COA) tidak memisahkan transaksi settled vs non-settled.

---

## Architecture & Tech Stack
- **SQL (SQLite):** Data modeling, indexing, **2-way reconciliation** (`LEFT JOIN` + `UNION ALL`), 14 customer-level fraud features, dan otomasi pemetaan COA.
- **Python (`pandas`, `scikit-learn`):** Data quality audit, *target leakage check*, 5-Fold Out-of-Fold Fraud Risk Scoring (`RandomForestClassifier`), evaluasi `ROC-AUC` & `PR-AUC`, dan ekstraksi *Feature Importance*.
- **Dashboard (`Streamlit`, `Plotly`):** Interactive threshold simulation untuk tim Risk Ops, visualisasi variance rekonsiliasi, eksposur nominal fraud per COA, serta export daftar audit (CSV).

---

## Project Structure
```text
lending-antifraud-recon/
├── sql/
│   ├── 01_reconciliation.sql   # Rekonsiliasi 2-arah (Approved vs Disbursement)
│   ├── 02_fraud_features.sql   # 14 fitur FPU & FTR tingkat nasabah
│   └── 03_coa_mapping.sql      # Pemetaan jurnal COA & filter Failed GL
├── pipeline_colab.py           # Pipeline ETL, SQL, Audit & ML Scoring
├── app.py                      # Streamlit + Plotly Dashboard
├── requirements.txt            # Dependencies untuk Streamlit Cloud
└── README.md
```

---

## Cara Menjalankan
1. Jalankan `pipeline_colab.py` (di Google Colab atau lokal dengan menyesuaikan `FOLDER_PATH`) untuk menghasilkan 4 file summary:
   - `summary_reconciliation.csv`
   - `summary_coa_mapping.csv`
   - `summary_customer_fraud_scores.csv`
   - `summary_feature_importance.csv`
2. Jalankan dashboard secara lokal:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Atau deploy langsung ke **Streamlit Community Cloud** dengan meng-upload `app.py`, `requirements.txt`, dan keempat file `summary_*.csv` ke GitHub (tanpa perlu `ngrok`).

---

## Key Insights & Model Evaluation
- **Fraud Risk Scoring (5-Fold Out-of-Fold Benchmark):**
  | Model Variant | ROC-AUC | PR-AUC | Precision (Fraud) | True Positive | False Negative | F1-Score (Fraud) |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **All 14 Features (Selected — Ongoing Risk Monitoring)** | **0.9135** | **0.7131** | 0.7343 | **246** | **120** | **0.7019** |
  | Pre-Decision 12 Features (Baseline — Without Rejection History) | 0.7673 | 0.5203 | **0.9128** | 178 | 188 | 0.6346 |

  Model **All 14 Features** dipilih untuk sistem *Ongoing Fraud Monitoring* karena menangkap **+68 kasus fraud tambahan** (246 vs 178 True Positive) dengan F1-score kelas minoritas lebih tinggi (`0.7019` vs `0.6346`).
- **Financial Reconciliation:** *(Isi dari hasil `df_recon`)* Ditemukan `X` kasus *unreconciled* dengan total selisih sebesar `...`, termasuk `Y` kasus *Unauthorized Disbursement*.
- **COA & Financial Exposure:** *(Isi dari hasil `df_coa_summary`)* Total nilai transaksi `Posted to GL` sebesar `...`, dengan `...` transaksi gagal dikecualikan dari GL, dan eksposur fraud terbesar berasal dari channel `...`.
