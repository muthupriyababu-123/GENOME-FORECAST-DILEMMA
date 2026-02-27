# 🌽 AI-Crop-Prediction

A Machine Learning project that predicts **maize (Zea mays) crop yield** using genomic SNP marker data and climate features (temperature & rainfall).

---

## 📁 Project Structure

```
AI-Crop-Prediction/
│
├── data/
│   ├── raw_dataset.csv         ← Original dataset with markers + climate + yield
│   └── processed_dataset.csv   ← Clean dataset (generated after preprocessing)
│
├── models/
│   └── trained_model.pkl       ← Saved Random Forest model (generated after training)
│
├── src/
│   ├── preprocessing.py        ← Data cleaning and train/test split
│   ├── train_model.py          ← Model training and evaluation
│   └── predict.py              ← Standalone prediction script
│
├── app.py                      ← Flask web application
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## 🧠 How It Works

| Input Feature | Description |
|---|---|
| Marker1, Marker2, Marker3 | SNP genomic markers (0=AA, 1=AT, 2=TT) |
| Temperature | Growing season temperature in °C |
| Rainfall | Seasonal rainfall in mm |
| **Yield** (output) | Predicted crop yield in **tons/hectare** |

The model is a **Random Forest Regressor** trained on 50 samples. It learns the relationship between genomic + climate features and crop yield.

---

## ⚙️ Installation

### 1. Clone or download the project

```bash
cd AI-Crop-Prediction
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Preprocess the data

```bash
python src/preprocessing.py
```

This will:
- Load `data/raw_dataset.csv`
- Handle missing values
- Save `data/processed_dataset.csv`

---

### Step 2 — Train the model

```bash
python src/train_model.py
```

This will:
- Train a Random Forest model
- Print R² Score and RMSE
- Save model to `models/trained_model.pkl`

---

### Step 3 — (Optional) Test a prediction from terminal

```bash
python src/predict.py
```

---

### Step 4 — Launch the web app

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 🌐 Web App Usage

1. Select **Marker1**, **Marker2**, **Marker3** values (0, 1, or 2)
2. Enter **Temperature** (e.g. 28 °C)
3. Enter **Rainfall** (e.g. 450 mm)
4. Click **Predict Yield**
5. The app shows predicted yield in **tons/hectare**

---

## 📊 Example Prediction

| Feature | Value |
|---|---|
| Marker1 | 1 |
| Marker2 | 2 |
| Marker3 | 0 |
| Temperature | 28 °C |
| Rainfall | 500 mm |
| **Predicted Yield** | **~6.5 t/ha** |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| pandas / numpy | Data processing |
| scikit-learn | Random Forest model |
| joblib | Model save/load |
| Flask | Web application |

---

## 👩‍🔬 Who Uses This?

This tool is designed for:
- **Plant breeders** working with genotyped maize lines
- **Agricultural researchers** analyzing genomic + climate interactions
- **Seed companies** screening varieties before costly field trials

> SNP marker values are obtained from genotyping labs (DNA sequencing output), not manually entered by farmers.

---

## 📜 License

This project is for educational purposes.
