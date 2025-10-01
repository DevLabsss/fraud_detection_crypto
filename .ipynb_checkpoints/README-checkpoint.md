# Fraud Detection — Everything Pack (Extended EDA + Notebook + Scripts)

Isi:

- `src/data_gen.py` — generator dataset synthetic kuat.
- `src/analysis_nb.py` — EDA (korelasi, distribusi fitur, label balance) + training Naive Bayes + threshold tuning.
- `policy_notes.md` — contoh kebijakan pencegahan.
- `data/transactions.csv` — dibuat otomatis saat run.
- `outputs/` — semua gambar EDA & evaluasi (heatmap, histogram, label balance, ROC, Confusion Matrix, metrics.json).
- `diagrams/flowchart.png` — flow proses.
- `Fraud_Detection_Everything.ipynb` — notebook lengkap (semua langkah + visual).

## Run (script)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/analysis_nb.py
```

## Run (notebook)

```bash
python -m pip install jupyterlab notebook
jupyter lab
```
