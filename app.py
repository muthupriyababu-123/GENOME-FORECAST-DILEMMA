# app.py
# -----------------------------------------------
# Flask web application for AI Crop Yield Prediction
# Features:
#   1. Manual input form
#   2. CSV lab report upload (AA/AT/TT encoded automatically)
#   3. SHAP explainability
#   4. Genomic marker recommendation
# Run: python app.py  →  Open: http://127.0.0.1:5000
# -----------------------------------------------

from flask import Flask, render_template_string, request
import joblib
import numpy as np
import pandas as pd
import io
import os

try:
    import shap
    SHAP_AVAILABLE = True
    print("[INFO] SHAP loaded successfully.")
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP not installed. Run: pip install shap")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB max upload

FEATURE_NAMES = ['Marker1', 'Marker2', 'Marker3', 'Temperature', 'Rainfall']
SNP_ENCODE    = {'AA': 0, 'AT': 1, 'TA': 1, 'TT': 2}
SNP_LABELS    = {0: '0 — AA (Homozygous Ref)', 1: '1 — AT (Heterozygous)', 2: '2 — TT (Homozygous Alt)'}
ICONS         = {'Marker1': '🧬', 'Marker2': '🧬', 'Marker3': '🧬', 'Temperature': '🌡️', 'Rainfall': '🌧️'}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'trained_model.pkl')
DATA_PATH  = os.path.join(os.path.dirname(__file__), 'data', 'processed_dataset.csv')

try:
    model = joblib.load(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
except FileNotFoundError:
    model = None
    print("[WARNING] Model not found. Run src/train_model.py first.")

explainer = None
if SHAP_AVAILABLE and model is not None:
    try:
        bg_data  = pd.read_csv(DATA_PATH)[FEATURE_NAMES]
        explainer = shap.TreeExplainer(model, bg_data)
        print("[INFO] SHAP explainer ready.")
    except Exception as e:
        print(f"[WARNING] SHAP explainer failed: {e}")


def encode_snp(val):
    """Convert AA/AT/TT string to 0/1/2. If already numeric, return as-is."""
    if isinstance(val, str):
        return SNP_ENCODE.get(val.strip().upper(), 0)
    return int(val)


def get_shap_values(input_df):
    if explainer is None:
        return None
    sv   = explainer(input_df)
    vals = sv.values[0] if len(sv.values.shape) == 2 else sv.values[0]
    return [round(float(v), 4) for v in vals]


def build_shap_and_markers(shap_vals, input_values):
    """Build shap_data and marker_summary from SHAP values."""
    base_yield = None
    shap_data  = None
    marker_summary = None

    if shap_vals is not None:
        result_val = sum(shap_vals)
        shap_data  = []
        for i, fname in enumerate(FEATURE_NAMES):
            shap_data.append({
                'name':      fname,
                'value':     round(shap_vals[i], 3),
                'abs_val':   abs(shap_vals[i]),
                'input_val': input_values[i],
                'icon':      ICONS.get(fname, '📊')
            })
        shap_data.sort(key=lambda x: x['abs_val'], reverse=True)

        marker_only = [d for d in shap_data if d['name'].startswith('Marker')]
        marker_only_sorted = sorted(marker_only, key=lambda x: x['value'], reverse=True)
        marker_summary = [{
            'name':      m['name'],
            'value':     m['value'],
            'abs_val':   m['abs_val'],
            'snp_label': SNP_LABELS.get(int(m['input_val']), str(m['input_val'])),
        } for m in marker_only_sorted]

    return shap_data, marker_summary


# =====================================================================
# HTML TEMPLATE
# =====================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>🌽 AI Crop Yield Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
            min-height: 100vh; display: flex;
            align-items: flex-start; justify-content: center; padding: 30px 20px;
        }
        .card {
            background: white; border-radius: 16px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
            padding: 36px; max-width: 600px; width: 100%;
        }
        h1 { color: #2e7d32; font-size: 1.8rem; margin-bottom: 4px; }
        .subtitle { color: #888; font-size: 0.88rem; margin-bottom: 20px; }

        /* Tabs */
        .tabs { display: flex; gap: 0; margin-bottom: 24px; border-radius: 10px; overflow: hidden; border: 1.5px solid #e0e0e0; }
        .tab-btn {
            flex: 1; padding: 11px; font-size: 0.9rem; font-weight: 700;
            background: #f9f9f9; border: none; cursor: pointer;
            color: #888; transition: all 0.2s;
        }
        .tab-btn.active { background: #2e7d32; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        label {
            display: block; font-weight: 600; color: #444;
            margin-bottom: 4px; margin-top: 12px; font-size: 0.85rem;
        }
        input, select {
            width: 100%; padding: 9px 13px;
            border: 1.5px solid #ddd; border-radius: 8px;
            font-size: 0.97rem; transition: border 0.2s;
        }
        input:focus, select:focus { outline: none; border-color: #43a047; }
        .hint { color: #aaa; font-size: 0.76rem; margin-top: 2px; }

        button[type=submit] {
            margin-top: 20px; width: 100%; padding: 12px;
            background: #2e7d32; color: white;
            font-size: 1rem; font-weight: 700;
            border: none; border-radius: 8px; cursor: pointer; transition: background 0.2s;
        }
        button[type=submit]:hover { background: #1b5e20; }

        .divider { border: none; border-top: 1px solid #eee; margin: 18px 0 8px; }
        .section-title {
            font-weight: 700; color: #666; font-size: 0.75rem;
            text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 2px;
        }

        /* Upload area */
        .upload-area {
            border: 2px dashed #a5d6a7; border-radius: 10px;
            padding: 30px; text-align: center;
            background: #f9fbe7; cursor: pointer; margin-top: 14px;
            transition: border 0.2s;
        }
        .upload-area:hover { border-color: #2e7d32; }
        .upload-icon { font-size: 2.5rem; margin-bottom: 8px; }
        .upload-text { font-size: 0.9rem; color: #555; font-weight: 600; }
        .upload-hint { font-size: 0.78rem; color: #aaa; margin-top: 4px; }
        .upload-area input[type=file] { display: none; }

        /* Sample CSV box */
        .sample-box {
            margin-top: 14px; background: #f5f5f5;
            border-radius: 8px; padding: 14px;
            font-size: 0.78rem; color: #555;
        }
        .sample-box strong { color: #333; display: block; margin-bottom: 6px; }
        .sample-box code {
            display: block; background: #eeeeee;
            padding: 8px 10px; border-radius: 6px;
            font-family: monospace; font-size: 0.78rem;
            white-space: pre; overflow-x: auto; line-height: 1.6;
        }

        /* Result */
        .result-box {
            margin-top: 22px; background: #e8f5e9;
            border-left: 5px solid #43a047;
            border-radius: 8px; padding: 16px 20px; text-align: center;
        }
        .yield-value { font-size: 2.4rem; font-weight: 800; color: #2e7d32; }
        .yield-label { color: #555; font-size: 0.88rem; margin-top: 2px; }

        /* Batch results table */
        .batch-table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.82rem; }
        .batch-table th {
            background: #2e7d32; color: white;
            padding: 8px 10px; text-align: left;
        }
        .batch-table td { padding: 8px 10px; border-bottom: 1px solid #eee; }
        .batch-table tr:nth-child(even) td { background: #f9f9f9; }
        .yield-cell { font-weight: 700; color: #2e7d32; }

        /* SHAP */
        .shap-box {
            margin-top: 18px; background: #fafafa;
            border: 1.5px solid #e0e0e0; border-radius: 10px; padding: 20px;
        }
        .shap-title { font-size: 1rem; font-weight: 700; color: #333; margin-bottom: 4px; }
        .shap-subtitle { font-size: 0.8rem; color: #888; margin-bottom: 14px; }
        .shap-row { margin-bottom: 12px; }
        .shap-feature-name {
            font-size: 0.85rem; font-weight: 600; color: #444;
            margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center;
        }
        .shap-badge { font-size: 0.78rem; font-weight: 700; padding: 2px 9px; border-radius: 20px; }
        .badge-pos { background: #e8f5e9; color: #2e7d32; }
        .badge-neg { background: #ffebee; color: #c62828; }
        .badge-neu { background: #f5f5f5; color: #777; }
        .bar-track { background: #eee; border-radius: 6px; height: 11px; position: relative; overflow: hidden; }
        .bar-center { position: absolute; left: 50%; top: 0; height: 100%; width: 2px; background: #ccc; z-index: 1; }
        .bar-fill { position: absolute; height: 100%; border-radius: 6px; }
        .bar-pos { background: linear-gradient(90deg, #81c784, #2e7d32); left: 50%; }
        .bar-neg { background: linear-gradient(90deg, #e53935, #ef9a9a); right: 50%; }
        .shap-legend { display: flex; gap: 16px; margin-top: 12px; font-size: 0.75rem; color: #888; }
        .leg-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 4px; vertical-align: middle; }
        .base-info { margin-top: 10px; font-size: 0.78rem; color: #aaa; text-align: center; }

        /* Marker box */
        .marker-box {
            margin-top: 18px; background: #fffde7;
            border: 1.5px solid #f9a825; border-radius: 10px; padding: 20px;
        }
        .marker-title { font-size: 1rem; font-weight: 700; color: #333; margin-bottom: 4px; }
        .marker-subtitle { font-size: 0.8rem; color: #888; margin-bottom: 14px; }
        .marker-row {
            display: flex; align-items: flex-start; gap: 12px;
            padding: 10px 12px; border-radius: 8px;
            margin-bottom: 8px; background: white; border: 1px solid #eee;
        }
        .marker-top { border: 2px solid #f9a825 !important; background: #fffde7 !important; }
        .marker-rank {
            width: 28px; height: 28px; border-radius: 50%;
            background: #f9a825; color: white; font-weight: 800; font-size: 0.85rem;
            display: flex; align-items: center; justify-content: center; flex-shrink: 0;
        }
        .marker-top .marker-rank { background: #f57f17; }
        .marker-info { flex: 1; }
        .marker-name { font-weight: 700; font-size: 0.9rem; color: #333; display: flex; align-items: center; gap: 8px; margin-bottom: 3px; }
        .best-badge { background: #fff8e1; color: #f57f17; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 20px; border: 1px solid #f9a825; }
        .marker-detail { font-size: 0.8rem; color: #666; }
        .recommendation-box { margin-top: 12px; padding: 12px 14px; background: #e8f5e9; border-radius: 8px; border-left: 4px solid #2e7d32; font-size: 0.84rem; color: #333; line-height: 1.5; }

        /* Error */
        .error-box {
            margin-top: 18px; background: #ffebee;
            border-left: 5px solid #e53935;
            border-radius: 8px; padding: 12px 16px; color: #c62828; font-size: 0.88rem;
        }
        .success-tag { background: #e8f5e9; color: #2e7d32; font-size: 0.78rem; font-weight: 700; padding: 2px 10px; border-radius: 20px; margin-left: 8px; }
    </style>
    <script>
        function switchTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById('tab-' + tab).classList.add('active');
            document.getElementById('content-' + tab).classList.add('active');
        }
        function triggerUpload() {
            document.getElementById('csv-file').click();
        }
        function showFileName() {
            var f = document.getElementById('csv-file').files[0];
            if (f) document.getElementById('file-name').innerText = '📄 ' + f.name;
        }
    </script>
</head>
<body>
<div class="card">
    <h1>🌽 AI Crop Predictor</h1>
    <p class="subtitle">Predict maize yield from genomic lab reports or manual input, with SHAP explanation.</p>

    <!-- Tabs -->
    <div class="tabs">
        <button class="tab-btn {% if active_tab != 'upload' %}active{% endif %}"
                id="tab-manual" onclick="switchTab('manual')">
            ✏️ Manual Input
        </button>
        <button class="tab-btn {% if active_tab == 'upload' %}active{% endif %}"
                id="tab-upload" onclick="switchTab('upload')">
            📂 Upload Lab Report (CSV)
        </button>
    </div>

    <!-- ===== TAB 1: MANUAL INPUT ===== -->
    <div class="tab-content {% if active_tab != 'upload' %}active{% endif %}" id="content-manual">
        <form method="POST" action="/" enctype="multipart/form-data">
            <input type="hidden" name="mode" value="manual">

            <p class="section-title">🧬 Genomic Markers (SNP Values)</p>
            <label>Marker 1</label>
            <select name="marker1">
                <option value="0" {% if form.marker1 == '0' %}selected{% endif %}>0 — AA (Homozygous Ref)</option>
                <option value="1" {% if form.marker1 == '1' %}selected{% endif %}>1 — AT (Heterozygous)</option>
                <option value="2" {% if form.marker1 == '2' %}selected{% endif %}>2 — TT (Homozygous Alt)</option>
            </select>
            <label>Marker 2</label>
            <select name="marker2">
                <option value="0" {% if form.marker2 == '0' %}selected{% endif %}>0 — AA</option>
                <option value="1" {% if form.marker2 == '1' %}selected{% endif %}>1 — AT</option>
                <option value="2" {% if form.marker2 == '2' %}selected{% endif %}>2 — TT</option>
            </select>
            <label>Marker 3</label>
            <select name="marker3">
                <option value="0" {% if form.marker3 == '0' %}selected{% endif %}>0 — AA</option>
                <option value="1" {% if form.marker3 == '1' %}selected{% endif %}>1 — AT</option>
                <option value="2" {% if form.marker3 == '2' %}selected{% endif %}>2 — TT</option>
            </select>

            <hr class="divider">
            <p class="section-title">🌦️ Climate Conditions</p>
            <label>Temperature (°C)</label>
            <input type="number" name="temperature" step="0.1" min="0" max="50"
                   placeholder="e.g. 28" value="{{ form.temperature or '' }}" required>
            <p class="hint">Typical range: 20 – 35 °C</p>
            <label>Rainfall (mm)</label>
            <input type="number" name="rainfall" step="1" min="0" max="2000"
                   placeholder="e.g. 450" value="{{ form.rainfall or '' }}" required>
            <p class="hint">Typical range: 300 – 700 mm</p>

            <button type="submit">🔍 Predict Yield + Explain</button>
        </form>
    </div>

    <!-- ===== TAB 2: CSV UPLOAD ===== -->
    <div class="tab-content {% if active_tab == 'upload' %}active{% endif %}" id="content-upload">
        <form method="POST" action="/" enctype="multipart/form-data">
            <input type="hidden" name="mode" value="upload">

            <div class="upload-area" onclick="triggerUpload()">
                <div class="upload-icon">🧬</div>
                <div class="upload-text">Click to upload Genotyping Lab Report (CSV)</div>
                <div class="upload-hint">Accepted columns: Sample_ID, SNP1, SNP2, SNP3, Temperature, Rainfall</div>
                <div id="file-name" style="margin-top:8px; color:#2e7d32; font-weight:600; font-size:0.85rem;"></div>
                <input type="file" id="csv-file" name="csv_file" accept=".csv" onchange="showFileName()">
            </div>

            <!-- Sample CSV format -->
            <div class="sample-box">
                <strong>📋 Expected CSV Format (from genotyping lab):</strong>
                <code>Sample_ID,SNP1,SNP2,SNP3,Temperature,Rainfall
Line_A,AA,AT,TT,28,500
Line_B,TT,AA,AT,30,450
Line_C,AT,TT,AA,25,600</code>
                <div style="margin-top:8px; color:#888;">
                    ✅ SNP values (AA/AT/TT) are <strong>automatically encoded</strong> by the system.<br>
                    ✅ Multiple samples can be predicted at once.
                </div>
            </div>

            <button type="submit">📊 Predict All Samples</button>
        </form>
    </div>

    {% if error %}
    <div class="error-box">⚠️ {{ error }}</div>
    {% endif %}

    <!-- ===== SINGLE PREDICTION RESULT (Manual) ===== -->
    {% if result is not none and not batch_results %}
    <div class="result-box">
        <div class="yield-value">{{ result }} t/ha</div>
        <div class="yield-label">Predicted Maize Yield (tons per hectare)</div>
    </div>

    {% if shap_data %}
    <div class="shap-box">
        <div class="shap-title">🧠 SHAP Feature Explanation</div>
        <div class="shap-subtitle">
            How much each feature <strong style="color:#2e7d32">increased ▲</strong>
            or <strong style="color:#c62828">decreased ▼</strong> the yield prediction.
        </div>
        {% set max_abs = shap_data | map(attribute='abs_val') | max %}
        {% for item in shap_data %}
        <div class="shap-row">
            <div class="shap-feature-name">
                <span>{{ item.icon }} {{ item.name }} <span style="color:#aaa;font-weight:400;">(input: {{ item.input_val }})</span></span>
                {% if item.value > 0.001 %}
                    <span class="shap-badge badge-pos">▲ +{{ item.value }} t/ha</span>
                {% elif item.value < -0.001 %}
                    <span class="shap-badge badge-neg">▼ {{ item.value }} t/ha</span>
                {% else %}
                    <span class="shap-badge badge-neu">≈ 0 t/ha</span>
                {% endif %}
            </div>
            <div class="bar-track">
                <div class="bar-center"></div>
                {% if item.value > 0 %}
                    <div class="bar-fill bar-pos" style="width: {{ [item.abs_val / max_abs * 48, 1] | max }}%;"></div>
                {% elif item.value < 0 %}
                    <div class="bar-fill bar-neg" style="width: {{ [item.abs_val / max_abs * 48, 1] | max }}%;"></div>
                {% endif %}
            </div>
        </div>
        {% endfor %}
        <div class="shap-legend">
            <span><span class="leg-dot" style="background:#2e7d32;"></span>Green = Increases yield</span>
            <span><span class="leg-dot" style="background:#e53935;"></span>Red = Decreases yield</span>
        </div>
        <p class="base-info">Base yield: <strong>{{ base_yield }} t/ha</strong> + SHAP = Final: <strong>{{ result }} t/ha</strong></p>
    </div>
    {% endif %}

    {% if marker_summary %}
    <div class="marker-box">
        <div class="marker-title">🧬 Genomic Marker Analysis</div>
        <div class="marker-subtitle">Which SNP marker contributes most to this prediction?</div>
        {% for m in marker_summary %}
        <div class="marker-row {% if loop.first %}marker-top{% endif %}">
            <div class="marker-rank">{{ loop.index }}</div>
            <div class="marker-info">
                <div class="marker-name">
                    {{ m.name }}
                    {% if loop.first and m.value > 0 %}<span class="best-badge">⭐ Most Feasible</span>
                    {% elif loop.first and m.value <= 0 %}<span class="best-badge" style="background:#ffebee;color:#c62828;border-color:#e53935;">⚠️ No Positive Marker</span>
                    {% endif %}
                </div>
                <div class="marker-detail">
                    SNP: <strong>{{ m.snp_label }}</strong> &nbsp;|&nbsp;
                    {% if m.value > 0 %}<span style="color:#2e7d32;font-weight:700;">▲ +{{ m.value }} t/ha (increases yield)</span>
                    {% elif m.value < 0 %}<span style="color:#c62828;font-weight:700;">▼ {{ m.value }} t/ha (decreases yield)</span>
                    {% else %}<span style="color:#888;">≈ No significant impact</span>{% endif %}
                </div>
            </div>
        </div>
        {% endfor %}
        <div class="recommendation-box">
            {% if marker_summary[0].value > 0 %}
            💡 <strong>Breeding Recommendation:</strong> Focus on <strong>{{ marker_summary[0].name }}</strong>
            ({{ marker_summary[0].snp_label }}) — it <strong style="color:#2e7d32">increases yield by +{{ marker_summary[0].value }} t/ha</strong>
            and is the most beneficial marker for this variety.
            {% else %}
            ⚠️ <strong>Note:</strong> All markers show neutral or negative impact. Try different SNP combinations.
            {% endif %}
        </div>
    </div>
    {% endif %}
    {% endif %}

    <!-- ===== BATCH RESULTS (CSV Upload) ===== -->
    {% if batch_results %}
    <div style="margin-top:22px;">
        <div style="font-size:1rem; font-weight:700; color:#333; margin-bottom:8px;">
            📊 Batch Prediction Results
            <span class="success-tag">✅ {{ batch_results|length }} samples processed</span>
        </div>
        <table class="batch-table">
            <thead>
                <tr>
                    <th>Sample ID</th>
                    <th>SNP1</th>
                    <th>SNP2</th>
                    <th>SNP3</th>
                    <th>Temp (°C)</th>
                    <th>Rainfall (mm)</th>
                    <th>Predicted Yield</th>
                </tr>
            </thead>
            <tbody>
                {% for row in batch_results %}
                <tr>
                    <td><strong>{{ row.Sample_ID }}</strong></td>
                    <td>{{ row.SNP1 }}</td>
                    <td>{{ row.SNP2 }}</td>
                    <td>{{ row.SNP3 }}</td>
                    <td>{{ row.Temperature }}</td>
                    <td>{{ row.Rainfall }}</td>
                    <td class="yield-cell">{{ row.Yield }} t/ha</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <p style="font-size:0.78rem; color:#aaa; margin-top:10px; text-align:center;">
            SNP values (AA/AT/TT) were automatically encoded for prediction.
        </p>
    </div>
    {% endif %}

</div>
</body>
</html>
"""


# =====================================================================
# ROUTES
# =====================================================================
@app.route("/", methods=["GET", "POST"])
def index():
    result         = None
    error          = None
    form           = {}
    shap_data      = None
    base_yield     = None
    marker_summary = None
    batch_results  = None
    active_tab     = "manual"

    if request.method == "POST":
        mode = request.form.get("mode", "manual")

        # ── MANUAL INPUT ──────────────────────────────────────
        if mode == "manual":
            try:
                form = {
                    "marker1":     request.form.get("marker1", "0"),
                    "marker2":     request.form.get("marker2", "0"),
                    "marker3":     request.form.get("marker3", "0"),
                    "temperature": request.form.get("temperature", ""),
                    "rainfall":    request.form.get("rainfall", ""),
                }
                marker1     = int(form["marker1"])
                marker2     = int(form["marker2"])
                marker3     = int(form["marker3"])
                temperature = float(form["temperature"])
                rainfall    = float(form["rainfall"])

                if model is None:
                    error = "Model not loaded. Run 'python src/train_model.py' first."
                else:
                    input_values = [marker1, marker2, marker3, temperature, rainfall]
                    input_df     = pd.DataFrame([input_values], columns=FEATURE_NAMES)
                    result       = round(float(model.predict(input_df)[0]), 2)

                    shap_vals = get_shap_values(input_df)
                    if shap_vals:
                        base_yield = round(float(result - sum(shap_vals)), 2)
                    shap_data, marker_summary = build_shap_and_markers(shap_vals, input_values)

            except ValueError:
                error = "Invalid input. Please enter valid numbers."
            except Exception as e:
                error = f"Prediction error: {str(e)}"

        # ── CSV UPLOAD ────────────────────────────────────────
        elif mode == "upload":
            active_tab = "upload"
            csv_file   = request.files.get("csv_file")

            if not csv_file or csv_file.filename == "":
                error = "No file uploaded. Please select a CSV file."
            else:
                try:
                    content = csv_file.read().decode("utf-8")
                    df      = pd.read_csv(io.StringIO(content))

                    # Flexible column mapping
                    col_map = {}
                    for col in df.columns:
                        cl = col.strip().lower()
                        if cl in ['snp1', 'marker1']: col_map[col] = 'Marker1'
                        elif cl in ['snp2', 'marker2']: col_map[col] = 'Marker2'
                        elif cl in ['snp3', 'marker3']: col_map[col] = 'Marker3'
                        elif cl in ['temperature', 'temp']: col_map[col] = 'Temperature'
                        elif cl in ['rainfall', 'rain']: col_map[col] = 'Rainfall'
                        elif cl in ['sample_id', 'sampleid', 'id', 'line']: col_map[col] = 'Sample_ID'
                    df.rename(columns=col_map, inplace=True)

                    required = ['Marker1', 'Marker2', 'Marker3', 'Temperature', 'Rainfall']
                    missing  = [c for c in required if c not in df.columns]
                    if missing:
                        error = f"Missing columns in CSV: {', '.join(missing)}. Check the format."
                    else:
                        if 'Sample_ID' not in df.columns:
                            df['Sample_ID'] = [f"Sample_{i+1}" for i in range(len(df))]

                        # Encode SNP values
                        for col in ['Marker1', 'Marker2', 'Marker3']:
                            df[col] = df[col].apply(encode_snp)

                        # Store original SNP labels for display
                        snp_display = {0: 'AA', 1: 'AT', 2: 'TT'}
                        df['SNP1'] = df['Marker1'].map(snp_display)
                        df['SNP2'] = df['Marker2'].map(snp_display)
                        df['SNP3'] = df['Marker3'].map(snp_display)

                        # Predict
                        X = df[FEATURE_NAMES]
                        df['Yield'] = model.predict(X).round(2)

                        batch_results = df[['Sample_ID','SNP1','SNP2','SNP3','Temperature','Rainfall','Yield']].to_dict('records')

                except Exception as e:
                    error = f"CSV processing error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        result=result, error=error, form=form,
        shap_data=shap_data, base_yield=base_yield,
        marker_summary=marker_summary,
        batch_results=batch_results,
        active_tab=active_tab
    )


if __name__ == "__main__":
    print("=" * 50)
    print("  🌽 AI Crop Yield Prediction App + SHAP")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True)
