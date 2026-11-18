# Lending Monitoring, Anti-Fraud (FPU/FTR) & Financial Reconciliation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lending-antifraud-recon.streamlit.app/)

**Live Interactive Dashboard:** https://lending-antifraud-recon.streamlit.app/

End-to-end **Data Analytics, Fraud Risk Scoring & Financial Control** portfolio project yang menggabungkan **SQL (SQLite)**, **Python (`pandas`, `scikit-learn`)**, dan **Interactive BI Dashboard (`Streamlit` + `Plotly`)** untuk mengaudit pencairan pinjaman, mendeteksi nasabah berisiko fraud, serta menjaga akurasi General Ledger (Chart of Accounts).

---

## Executive Summary (Key Findings)

Berdasarkan analisis terhadap **18,314 nasabah** (*Loan Applications / FPU*) dan **50,000 transaksi** (*Transactions / FTR*):

| Area Analisis | Metrik Utama | Temuan Kunci |
| :--- | :--- | :--- |
| **Anti-Fraud Risk Scoring** | **18,314 nasabah** (Fraud Rate: **2.00% / 366 kasus**) | Model **All 14 Features** mencapai **ROC-AUC 0.9135** & **PR-AUC 0.7131**, mengisolasi **220 nasabah High Risk (1.20%)** untuk blokir/audit prioritas. |
| **Two-Way Reconciliation** | **17,806 nasabah** dalam cakupan audit pencairan | Rekonsiliasi 2-arah mendeteksi **651 kasus Unauthorized Disbursement** (pencairan tanpa approval) senilai **3,685,707.13** dan **18 Orphan Failed Disbursement**. |
| **COA & GL Audit Control** | **50,000 transaksi** (Total Nominal: **250,283,716.46**) | **255 transaksi gagal** senilai **2,207,064.06** berhasil dikecualikan dari GL (**100% terindikasi fraud**), dan **4,995 transaksi `EMI Payment`** (`24,418,610.74`) terdeteksi sebagai *Unmapped Account*. |

---

## Business Problem
Perusahaan *fintech lending* menghadapi 3 tantangan kontrol operasional dan risiko utama:
1. **Disbursement Leakage & Reconciliation Gaps:** Ketidaksesuaian antara pengajuan pinjaman yang disetujui (*FPU / Loan Applications*) dengan realisasi pencairan dana (*FTR / Transactions*), termasuk risiko pencairan tanpa persetujuan (*Unauthorized Disbursement*) maupun kekurangan pencairan.
2. **Multi-Channel Fraud Prioritization:** Sulitnya memprioritaskan nasabah berisiko tinggi karena sinyal fraud tersebar pada profil pengajuan kredit (*CIBIL score*, riwayat penolakan, DTI) dan perilaku transaksi (transaksi gagal, lonjakan nominal, perangkat/IP).
3. **General Ledger (GL) Audit Accuracy:** Risiko transaksi berstatus `Failed` ikut tercatat ke jurnal akuntansi dan adanya tipe transaksi baru yang belum terpetakan pada Chart of Accounts (COA).

---

## Architecture & Tech Stack
- **SQL (SQLite — [`sql/`](sql/)):** Data modeling, indexing pada `customer_id`, **Two-Way Reconciliation** (`LEFT JOIN` + `UNION ALL`), rekayasa **14 fitur fraud** tingkat nasabah, serta otomasi pemetaan jurnal COA.
- **Python ([`pipeline_colab.py`](pipeline_colab.py)):** Data quality audit (validasi duplikat & missing values), *target leakage benchmark*, **5-Fold Out-of-Fold Fraud Risk Scoring** (`RandomForestClassifier` dengan `class_weight="balanced"`), evaluasi `ROC-AUC` & `PR-AUC`, dan ekstraksi *Feature Importance*.
- **Interactive Dashboard ([`app.py`](app.py)):** Dashboard **Streamlit + Plotly** dengan *Control Panel* slider batas risiko dinamis, grafik distribusi skor & *Top Fraud Drivers*, grafik *variance* rekonsiliasi, eksposur nominal fraud per COA, serta tombol **Export CSV** untuk tindak lanjut tim Ops/Audit.

---

## Detailed Key Insights & Model Evaluation

### 1. Anti-Fraud Model Evaluation (5-Fold Out-of-Fold Benchmark)
Untuk memastikan hasil evaluasi yang jujur dan memeriksa dampak fitur historis penolakan (`f2_declined_apps_count` & `f3_rejection_rate`), dilakukan perbandingan dua skenario model menggunakan *5-Fold Stratified Out-of-Fold Cross-Validation*:

| Model Variant | ROC-AUC | PR-AUC | Precision (Fraud) | True Positive | False Negative | F1-Score (Fraud) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All 14 Features (Selected — Ongoing Risk Monitoring)** | **0.9135** | **0.7131** | 0.7343 | **246** | **120** | **0.7019** |
| Pre-Decision 12 Features (Baseline — Without Rejection History) | 0.7673 | 0.5203 | **0.9128** | 178 | 188 | 0.6346 |

- **Alasan Pemilihan Model:** Model **All 14 Features** dipilih untuk skenario *Ongoing Account Monitoring* karena menangkap **+68 kasus fraud tambahan** (246 vs 178 *True Positive*) dan menurunkan *False Negative* dari 188 menjadi 120, dengan F1-score kelas minoritas lebih unggul (`0.7019` vs `0.6346`). Model 12 fitur dipertahankan sebagai *baseline* untuk evaluasi nasabah baru (*first-time applicant*) yang belum memiliki riwayat penolakan.
- **Segmentasi Risiko Nasabah (Rata-rata Risk Score: 19.96):**
  - **High Risk (`Score > 70` — Block/Audit):** **220 nasabah** (`1.20%`)
  - **Medium Risk (`Score 30–70` — Manual Review):** **5,537 nasabah** (`30.23%`)
  - **Low Risk (`Score <= 30` — Auto-Pass):** **12,557 nasabah** (`68.57%`)
- **Top 4 Faktor Pendorong Risiko Fraud (Menyumbang 82.00% Daya Prediksi):**
  1. `f10_failed_trx_count` (**35.30%**) — konsisten dengan temuan audit bahwa 100% transaksi gagal berstatus fraud.
  2. `f5_min_cibil_score` (**17.15%**) — skor kredit terendah nasabah.
  3. `f2_declined_apps_count` (**15.04%**) — jumlah pengajuan pinjaman yang ditolak.
  4. `f3_rejection_rate` (**14.51%**) — rasio penolakan terhadap total pengajuan nasabah.

---

### 2. Two-Way Lending & Disbursement Reconciliation
Total nilai pinjaman `Approved` tercatat sebesar **21,038,929,000.00**, sedangkan pencairan `Loan Disbursement` berstatus `Success` pada periode transaksi tercatat sebesar **25,223,307.13** (ditambah `Failed` sebesar **149,158.08**):

| Status Rekonsiliasi | Jumlah Nasabah | Net Variance | Absolut Variance | Risiko / Tindak Lanjut |
| :--- | :---: | :---: | :---: | :--- |
| **Unrecorded Disbursement (Belum Cair di Trx)** | 13,325 | `16,288,037,000.00` | `16,288,037,000.00` | Verifikasi jadwal pencairan / *cut-off* periode data |
| **Under-Disbursement (Selisih Kurang Cair)** | 3,812 | `4,729,354,400.00` | `4,729,354,400.00` | Audit pencairan parsial atau perbedaan satuan/batch |
| **Unauthorized Disbursement (Cair Tanpa Approval)** | **651** | **`-3,685,707.13`** | **`3,685,707.13`** | **Prioritas Audit Tinggi — indikasi bypass kontrol approval** |
| **Orphan Failed Disbursement (Gagal Tanpa Approval)** | **18** | `0.00` | `0.00` | Investigasi percobaan pencairan ilegal yang gagal |

- **Total Absolut Variance:** **21,021,077,107.13** (Net Variance: **21,013,705,692.87**).
- **Nilai Tambah Kontrol 2-Arah:** Tanpa metode `LEFT JOIN + UNION ALL`, sebanyak **669 kasus anomali** (`651` pencairan tanpa approval senilai `3,685,707.13` dan `18` percobaan gagal tanpa approval) akan luput dari rekonsiliasi satu arah biasa.

---

### 3. Automated COA Journal Mapping & Fraud Financial Exposure
Dari total **50,000 transaksi** (total nilai **250,283,716.46**):
- **Posted to GL (`Success`):** **49,745 transaksi** senilai **248,076,652.40**.
  - Di dalamnya masih terdapat **247 transaksi fraud lolos** senilai **1,937,852.40** yang merepresentasikan potensi kerugian finansial aktual.
- **Excluded from GL (`Failed`):** **255 transaksi** senilai **2,207,064.06** berhasil dikecualikan agar tidak mengotori General Ledger.
  - **Anomali Kritis:** **100% transaksi gagal (255 dari 255 transaksi)** ditandai sebagai `fraud_flag = 1`, memperkuat aturan bahwa setiap lonjakan *failed transactions* harus memicu eskalasi *fraud alert* otomatis.
- **Total Eksposur Fraud Keseluruhan:** **502 transaksi** senilai **4,144,916.46**.
  - **Top 3 Channel Berdasarkan Eksposur Nominal Fraud:**
    1. **Bill Payment:** `525,717.87` (46 transaksi)
    2. **Credit Card:** `497,318.55` (48 transaksi)
    3. **Fund Transfer:** `479,788.37` (49 transaksi)
  - **Top Channel Berdasarkan Frekuensi Fraud:** **UPI** (60 transaksi | `464,331.25`) dan **EMI Payment** (56 transaksi | `425,215.95`).
- **Temuan Audit Akuntansi (`Unmapped Account Alert`):** Kontrol otomatis mendeteksi **4,995 transaksi `EMI Payment` (`Success`)** senilai **24,418,610.74** belum memiliki pemetaan akun GL.

---

## Business Recommendations
1. **Investigasi Segera 651 Kasus `Unauthorized Disbursement`:** Lakukan *cross-check* antara *core lending system* dan *payment gateway* terhadap pencairan senilai `3,685,707.13` yang tidak memiliki record `Approved`, serta terapkan validasi `application_id` wajib sebelum eksekusi pencairan.
2. **Real-Time Alert untuk `Failed Transactions`:** Karena 100% transaksi `Failed` di dataset ini berkorelasi dengan fraud dan `f10_failed_trx_count` menyumbang 35.30% *feature importance*, terapkan aturan *velocity block* otomatis bagi nasabah dengan percobaan transaksi gagal berulang, serta audit prioritas pada **220 nasabah High Risk**.
3. **Pembaruan Master COA untuk `EMI Payment`:** Tambahkan aturan pemetaan GL untuk transaksi `EMI Payment` (`Dr 1-1100 Kas & Bank / Cr 1-1300 Piutang Pinjaman`) guna menutup celah `Unmapped Account` senilai `24,418,610.74`.

---

## Project Structure
```text
lending-antifraud-recon/
├── sql/
│   ├── 01_reconciliation.sql          # Rekonsiliasi 2-arah (Approved vs Disbursement)
│   ├── 02_fraud_features.sql          # 14 fitur FPU & FTR tingkat nasabah
│   └── 03_coa_mapping.sql             # Pemetaan jurnal COA & filter Failed GL
├── pipeline_colab.py                  # Pipeline ETL, SQL, Quality Audit & 5-Fold OOF Scoring
├── app.py                             # Interactive Streamlit + Plotly Dashboard
├── summary_reconciliation.csv         # Data mart hasil rekonsiliasi 2-arah
├── summary_coa_mapping.csv            # Data mart pemetaan COA & eksposur fraud
├── summary_customer_fraud_scores.csv  # Data mart skor risiko 18,314 nasabah
├── summary_feature_importance.csv     # Bobot 14 fitur pendorong risiko fraud
├── requirements.txt                   # Dependencies untuk Streamlit Cloud
└── README.md
```

---

## Cara Menjalankan Secara Lokal
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan dashboard Streamlit:
   ```bash
   streamlit run app.py
   ```
3. *(Opsional)* Untuk menjalankan ulang seluruh proses SQL & ML dari dataset mentah, sesuaikan `FOLDER_PATH` pada [pipeline_colab.py](pipeline_colab.py) lalu eksekusi script tersebut.
