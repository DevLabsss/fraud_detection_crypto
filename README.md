# 🛡️ Fraud Detection — Cryptocurrency (Naive Bayes Baseline)

Kelompok 1 — Universitas Pamulang  
📚 Mata Kuliah: Data Mining  
👨‍🏫 Dosen: Tri Prasetyo  

## 📌 Judul
Prediksi Transaksi Fraud pada Cryptocurrency Menggunakan Algoritma Naive Bayes

## 👥 Anggota Kelompok
- Achmad Syahril Fauzi (231011450396)  
- Abdul Fakhry (231011450644)  
- Ahmad Imam (231011450458)  

---

## 🎯 Tujuan
- Membangun sistem sederhana untuk klasifikasi normal vs fraud.  
- Menyusun baseline model dengan Naive Bayes.  
- Menunjukkan bagaimana machine learning membantu keamanan transaksi digital.  

---

## 📊 Dataset
- Jenis: Synthetic dataset (8.000 baris, dibuat otomatis dengan Python).  
- Fitur (X):  
  - `amount` — jumlah transaksi  
  - `transaction_freq_24h` — frekuensi transaksi 24 jam  
  - `account_age_days` — umur akun (hari)  
  - `is_weekend` — 0/1 apakah transaksi terjadi saat weekend  
  - `risk_score` — skor risiko gabungan  
- Label (y): `is_fraud` — 0 (normal) / 1 (fraud)  
- Dataset disimpan di `data/transactions.csv`, otomatis dibuat jika belum ada.  

---

## 🛠️ Teknologi
- Python 3.9+  
- scikit-learn  
- pandas, numpy  
- matplotlib  
- Jupyter Notebook  

---

## 📂 Struktur Repo
```
fraud_detection_crypto/
├── README.md
├── requirements.txt
├── policy_notes.md
├── data/
│   └── transactions.csv
├── outputs/
│   ├── corr_heatmap.png
│   ├── distribution_amount.png
│   ├── distribution_tx_freq.png
│   ├── distribution_account_age.png
│   ├── distribution_risk_score.png
│   ├── label_balance.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── metrics.json
├── diagrams/
│   └── flowchart.png
├── src/
│   ├── data_gen.py
│   └── analysis_nb.py
└── Fraud_Detection_Everything.ipynb
```

---

## ⚙️ Cara Menjalankan

### 1. Clone Repo
```bash
git clone https://github.com/DevLabsss/fraud_detection_crypto.git
cd fraud_detection_crypto
```

### 2. Setup Virtual Environment
**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Dataset
Jika `data/transactions.csv` hilang, buat ulang dengan:
```bash
python -m src.data_gen
```

### 5. Jalankan Analisis
**Via Script**
```bash
python -m src.analysis_nb
```
Output model & grafik akan muncul di folder `outputs/`.

**Via Jupyter Notebook**
```bash
pip install notebook jupyterlab ipykernel
jupyter lab
```
Lalu buka file **Fraud_Detection_Everything.ipynb** → Run All Cells.  

### 6. Hasil Output
- 📊 Grafik distribusi & korelasi (`outputs/`)  
- 🧾 Metrics model (`metrics.json`)  
- 🖼️ Confusion matrix & ROC curve  

---

## 📈 Evaluasi
- Accuracy: 95.13%  
- Precision: 85.84%  
- Recall: 93.17%  
- F1-score: 89.35%  
- AUC: 95.82%  
- Visualisasi: Confusion Matrix & ROC Curve  

---

## 🛡️ Kebijakan (Policy)
- Transaksi besar & akun muda → REVIEW / 2FA.  
- Transaksi ≥7 kali/24 jam → WATCHLIST otomatis.  
- Weekend + nominal besar → manual verify.  
- Probabilitas fraud ≥0.85 → BLOCK & KYC ulang.  

---

## 🚀 Next Step
- Uji dataset publik (Ethereum Fraud Detection di Kaggle).  
- Tambahkan balancing (SMOTE/undersampling) jika dataset sangat imbalance.  
- Bandingkan dengan model lain (LogReg, Random Forest).  

---

## 📝 Lisensi
Untuk kebutuhan akademik/pembelajaran.  
