# streamlit_app.py
# =====================================================================
# 🌽 AI Crop Predictor - Streamlit Version
# Predicts maize yield from genotyping lab reports using XGBoost + SHAP
# Run: streamlit run streamlit_app.py
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Try imports ---
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="🌽 AI Crop Predictor",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(160deg, #f0faf0 0%, #e8f5e9 50%, #f9fbe7 100%);
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
        border-radius: 20px;
        padding: 40px 48px;
        margin-bottom: 32px;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '🌽';
        position: absolute;
        right: 40px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 6rem;
        opacity: 0.15;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        color: white !important;
    }
    .hero p {
        font-size: 1rem;
        opacity: 0.85;
        margin: 0;
        color: white !important;
    }

    /* Section cards */
    .section-card {
        background: white;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
        border: 1px solid #e8f5e9;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-subtitle {
        font-size: 0.82rem;
        color: #888;
        margin-bottom: 18px;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin: 16px 0;
    }
    .metric-card {
        flex: 1;
        min-width: 140px;
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #2e7d32;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1b5e20;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #666;
        margin-top: 2px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Yield badge */
    .yield-high { color: #1b5e20; font-weight: 800; }
    .yield-mid  { color: #f57f17; font-weight: 800; }
    .yield-low  { color: #c62828; font-weight: 800; }

    /* Step badges */
    .step-badge {
        display: inline-block;
        background: #2e7d32;
        color: white;
        border-radius: 50%;
        width: 28px; height: 28px;
        text-align: center;
        line-height: 28px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 8px;
    }

    /* Info box */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #1565c0;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.83rem;
        color: #0d47a1;
        margin: 12px 0;
    }

    /* Warning box */
    .warn-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.83rem;
        color: #e65100;
        margin: 12px 0;
    }

    /* Success box */
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.83rem;
        color: #1b5e20;
        margin: 12px 0;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32, #1b5e20) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 14px 32px !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(46,125,50,0.4) !important;
    }

    /* Dataframe styling */
    .dataframe { border-radius: 10px !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b5e20, #2e7d32) !important;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stMarkdown h3 { color: #a5d6a7 !important; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CONSTANTS
# =====================================================================
FEATURE_NAMES = ['Marker1', 'Marker2', 'Marker3', 'Temperature', 'Rainfall']
SNP_ENCODE    = {'AA': 0, 'AT': 1, 'TA': 1, 'TT': 2}
SNP_DECODE    = {0: 'AA', 1: 'AT', 2: 'TT'}
MODEL_PATH    = os.path.join(os.path.dirname(__file__), 'models', 'trained_model.pkl')
DATA_PATH     = os.path.join(os.path.dirname(__file__), 'data', 'processed_dataset.csv')


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def encode_snp(val):
    """Convert AA/AT/TT to 0/1/2. If already numeric, return as-is."""
    if isinstance(val, str):
        return SNP_ENCODE.get(val.strip().upper(), 0)
    return int(val)


@st.cache_resource
def load_model():
    """Load trained model (cached so it loads only once)."""
    # Try XGBoost JSON model first
    xgb_path = os.path.join(os.path.dirname(__file__), 'maize_yield_model.json')
    if XGB_AVAILABLE and os.path.exists(xgb_path):
        m = xgb.XGBRegressor()
        m.load_model(xgb_path)
        return m, "XGBoost"
    # Fall back to sklearn pkl model
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH), "Random Forest"
    return None, None


@st.cache_resource
def load_explainer(_model):
    """Build SHAP TreeExplainer (cached)."""
    if not SHAP_AVAILABLE or _model is None:
        return None
    try:
        bg = pd.read_csv(DATA_PATH)[FEATURE_NAMES]
        return shap.TreeExplainer(_model, bg)
    except Exception:
        try:
            return shap.TreeExplainer(_model)
        except Exception:
            return None


def process_csv(uploaded_file, temperature, rainfall):
    """
    Read uploaded CSV, encode SNP values,
    add climate data, return ready DataFrame.
    """
    content = uploaded_file.read().decode("utf-8")
    df      = pd.read_csv(io.StringIO(content))

    # Flexible column name mapping
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if   cl in ['snp1', 'marker1']: col_map[col] = 'Marker1'
        elif cl in ['snp2', 'marker2']: col_map[col] = 'Marker2'
        elif cl in ['snp3', 'marker3']: col_map[col] = 'Marker3'
        elif cl in ['sample_id', 'sampleid', 'id', 'line', 'variety']: col_map[col] = 'Sample_ID'
    df.rename(columns=col_map, inplace=True)

    if 'Sample_ID' not in df.columns:
        df['Sample_ID'] = [f"Sample_{i+1}" for i in range(len(df))]

    # Store original SNP strings for display
    for src, dst in [('Marker1','SNP1_orig'), ('Marker2','SNP2_orig'), ('Marker3','SNP3_orig')]:
        if src in df.columns:
            df[dst] = df[src].astype(str)

    # Encode SNP values
    for col in ['Marker1', 'Marker2', 'Marker3']:
        if col in df.columns:
            df[col] = df[col].apply(encode_snp)

    # Add climate data (same for all samples from slider)
    df['Temperature'] = temperature
    df['Rainfall']    = rainfall

    return df


def make_shap_plot(explainer, X_df):
    """Generate SHAP summary bar plot and return matplotlib figure."""
    sv = explainer(X_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    # Mean absolute SHAP per feature
    mean_shap = np.abs(sv.values).mean(axis=0)
    feature_importance = pd.Series(mean_shap, index=FEATURE_NAMES).sort_values(ascending=True)

    colors = ['#ef5350' if v < 0 else '#66bb6a' for v in feature_importance.values]
    colors = ['#43a047'] * len(feature_importance)  # all green for mean abs

    bars = ax.barh(feature_importance.index, feature_importance.values,
                   color=colors, height=0.55, edgecolor='white', linewidth=1.5)

    # Value labels
    for bar, val in zip(bars, feature_importance.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left',
                fontsize=9, fontweight='600', color='#333')

    ax.set_xlabel('Mean |SHAP Value| — Average Impact on Yield (t/ha)',
                  fontsize=9, color='#555')
    ax.set_title('Feature Importance (SHAP)', fontsize=12,
                 fontweight='800', color='#1b5e20', pad=12)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(colors='#555', labelsize=9)
    ax.xaxis.set_tick_params(length=0)
    plt.tight_layout()
    return fig


def make_individual_shap_plot(shap_values, feature_names, sample_id):
    """Waterfall-style bar chart for a single sample."""
    vals  = shap_values
    pairs = sorted(zip(feature_names, vals), key=lambda x: x[1])

    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = ['#ef5350' if v < 0 else '#43a047' for v in values]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars = ax.barh(labels, values, color=colors, height=0.5,
                   edgecolor='white', linewidth=1)

    for bar, val in zip(bars, values):
        x_pos = val + (0.003 if val >= 0 else -0.003)
        ha    = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center', ha=ha,
                fontsize=8.5, fontweight='600',
                color='#2e7d32' if val >= 0 else '#c62828')

    ax.axvline(0, color='#999', linewidth=1, linestyle='--')
    ax.set_xlabel('SHAP Value (impact on yield t/ha)', fontsize=8.5, color='#666')
    ax.set_title(f'Feature Contributions — {sample_id}',
                 fontsize=10, fontweight='700', color='#1b5e20')
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(colors='#555', labelsize=8.5)
    plt.tight_layout()
    return fig


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("## 🌽 AI Crop Predictor")
    st.markdown("---")
    st.markdown("### 📖 How It Works")
    st.markdown("""
1. **Upload** genotyping lab CSV
2. **Set** climate conditions
3. **Predict** yield instantly
4. **Explore** SHAP explanations
    """)
    st.markdown("---")
    st.markdown("### 📋 CSV Format")
    st.markdown("""
```
Sample_ID,SNP1,SNP2,SNP3
Line_A,AA,AT,TT
Line_B,TT,AA,AT
```
SNP values: `AA`, `AT`, `TT`
    """)
    st.markdown("---")
    st.markdown("### 🧬 SNP Encoding")
    st.markdown("""
| SNP | Code |
|-----|------|
| AA  |  0   |
| AT  |  1   |
| TT  |  2   |
    """)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("Built with Random Forest + SHAP for explainable crop yield prediction.")


# =====================================================================
# MAIN PAGE
# =====================================================================

# Hero Header
st.markdown("""
<div class="hero">
    <h1>🌽 AI Crop Predictor</h1>
    <p>Upload your genotyping lab report → Get instant yield predictions + SHAP explanations</p>
</div>
""", unsafe_allow_html=True)

# Load model
model, model_type = load_model()

if model is None:
    st.error("⚠️ Model not found! Please run `python src/train_model.py` first.")
    st.stop()

st.success(f"✅ Model loaded: **{model_type}**")

# =====================================================================
# STEP 1 — FILE UPLOAD
# =====================================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📂 Step 1 — Upload Genotyping Lab Report</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Upload a CSV file from your genotyping lab. SNP values (AA/AT/TT) are encoded automatically.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose your lab report CSV file",
    type=["csv"],
    help="CSV must have columns: Sample_ID, SNP1 (or Marker1), SNP2, SNP3"
)

if uploaded_file:
    st.markdown('<div class="success-box">✅ File uploaded successfully! Scroll down to set climate data and predict.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="info-box">💡 No file yet? Use the sample format shown in the sidebar. You can download a sample from the <b>data/</b> folder.</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# STEP 2 — CLIMATE CONDITIONS
# =====================================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌦️ Step 2 — Set Climate Conditions</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">These climate values will be applied to all samples in your uploaded file.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider(
        "🌡️ Temperature (°C)",
        min_value=15.0, max_value=45.0,
        value=28.0, step=0.5,
        help="Growing season average temperature"
    )
    st.caption(f"Selected: **{temperature}°C**")

with col2:
    rainfall = st.slider(
        "🌧️ Rainfall (mm)",
        min_value=200, max_value=900,
        value=500, step=10,
        help="Seasonal total rainfall in mm"
    )
    st.caption(f"Selected: **{rainfall} mm**")

# Quick climate summary
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">{temperature}°C</div>
        <div class="metric-label">Temperature</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{rainfall}mm</div>
        <div class="metric-label">Rainfall</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{'🔴 High' if temperature > 32 else '🟡 Moderate' if temperature > 28 else '🟢 Optimal'}</div>
        <div class="metric-label">Heat Stress</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{'🟢 Good' if rainfall > 500 else '🟡 Low' if rainfall > 350 else '🔴 Dry'}</div>
        <div class="metric-label">Water Availability</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# STEP 3 — PREDICT BUTTON
# =====================================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔍 Step 3 — Predict Yield</div>', unsafe_allow_html=True)

predict_clicked = st.button("🌽 Predict Yield for All Samples", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# PREDICTION + OUTPUT
# =====================================================================
if predict_clicked:
    if uploaded_file is None:
        st.error("⚠️ Please upload a CSV file first!")
    else:
        with st.spinner("🔄 Processing lab report and predicting yields..."):
            try:
                # Reset file pointer (in case it was read before)
                uploaded_file.seek(0)

                # Process CSV
                df = process_csv(uploaded_file, temperature, rainfall)

                # Check required columns
                missing = [c for c in FEATURE_NAMES if c not in df.columns]
                if missing:
                    st.error(f"❌ Missing columns: {', '.join(missing)}. Check your CSV format.")
                    st.stop()

                # Feature matrix
                X = df[FEATURE_NAMES].copy()

                # Predict yield
                df['Predicted_Yield'] = model.predict(X).round(2)

                # ── STEP 4: OUTPUT TABLE ──────────────────────────────
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📊 Step 4 — Prediction Results</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="section-subtitle">{len(df)} samples processed successfully.</div>', unsafe_allow_html=True)

                # Summary metrics
                avg_yield = df['Predicted_Yield'].mean()
                max_yield = df['Predicted_Yield'].max()
                min_yield = df['Predicted_Yield'].min()
                best_line = df.loc[df['Predicted_Yield'].idxmax(), 'Sample_ID']

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-value">{avg_yield:.2f}</div>
                        <div class="metric-label">Avg Yield (t/ha)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color:#1b5e20">{max_yield:.2f}</div>
                        <div class="metric-label">Best Yield (t/ha)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color:#c62828">{min_yield:.2f}</div>
                        <div class="metric-label">Lowest Yield (t/ha)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1rem">{best_line}</div>
                        <div class="metric-label">Best Variety</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Build display table
                display_cols = ['Sample_ID']
                for orig, label in [('SNP1_orig','SNP1'), ('SNP2_orig','SNP2'), ('SNP3_orig','SNP3')]:
                    if orig in df.columns:
                        df[label] = df[orig]
                        display_cols.append(label)

                display_cols += ['Temperature', 'Rainfall', 'Predicted_Yield']
                display_df = df[display_cols].copy()
                display_df.columns = [c.replace('Predicted_Yield', 'Yield (t/ha)') for c in display_df.columns]

                st.dataframe(
                    display_df.style.background_gradient(
                        subset=['Yield (t/ha)'],
                        cmap='Greens'
                    ).format({'Yield (t/ha)': '{:.2f}'}),
                    use_container_width=True,
                    height=min(400, 60 + len(df) * 38)
                )

                # Download button
                csv_out = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_out,
                    file_name="yield_predictions.csv",
                    mime="text/csv"
                )

                st.markdown('</div>', unsafe_allow_html=True)

                # ── STEP 5: SHAP EXPLANATION ──────────────────────────
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🧠 Step 5 — SHAP Feature Explanation</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-subtitle">SHAP (SHapley Additive Explanations) shows how much each feature increased or decreased the yield prediction.</div>', unsafe_allow_html=True)

                if not SHAP_AVAILABLE:
                    st.markdown('<div class="warn-box">⚠️ SHAP not installed. Run: <code>pip install shap</code></div>', unsafe_allow_html=True)
                else:
                    explainer = load_explainer(model)
                    if explainer is None:
                        st.warning("SHAP explainer could not be built.")
                    else:
                        sv = explainer(X)

                        tab1, tab2 = st.tabs(["📊 Overall Feature Importance", "🔍 Per-Sample Analysis"])

                        with tab1:
                            st.markdown("**Which features matter most across all samples?**")
                            fig_summary = make_shap_plot(explainer, X)
                            st.pyplot(fig_summary, use_container_width=True)
                            plt.close()

                            # Text summary
                            mean_shap = np.abs(sv.values).mean(axis=0)
                            top_idx   = np.argmax(mean_shap)
                            top_feat  = FEATURE_NAMES[top_idx]
                            st.markdown(f"""
                            <div class="success-box">
                            🏆 <strong>Most Influential Feature: {top_feat}</strong><br>
                            On average, <strong>{top_feat}</strong> has the highest impact on yield prediction
                            with a mean |SHAP| value of <strong>{mean_shap[top_idx]:.3f} t/ha</strong>.
                            </div>
                            """, unsafe_allow_html=True)

                        with tab2:
                            st.markdown("**Select a sample to see its individual SHAP breakdown:**")
                            selected = st.selectbox(
                                "Choose sample",
                                df['Sample_ID'].tolist(),
                                key="shap_sample"
                            )
                            idx        = df['Sample_ID'].tolist().index(selected)
                            shap_vals  = sv.values[idx]
                            input_vals = X.iloc[idx].tolist()

                            # Individual SHAP bar chart
                            fig_ind = make_individual_shap_plot(shap_vals, FEATURE_NAMES, selected)
                            st.pyplot(fig_ind, use_container_width=True)
                            plt.close()

                            # Feature breakdown table
                            breakdown = pd.DataFrame({
                                'Feature':     FEATURE_NAMES,
                                'Input Value': [f"{v:.1f}" for v in input_vals],
                                'SHAP Impact': [f"{'+' if v > 0 else ''}{v:.4f} t/ha" for v in shap_vals],
                                'Effect':      ['▲ Increases yield' if v > 0.001 else '▼ Decreases yield' if v < -0.001 else '≈ Neutral' for v in shap_vals]
                            })
                            st.dataframe(breakdown, use_container_width=True, hide_index=True)

                            # Best marker recommendation
                            marker_shaps = [(f, v) for f, v in zip(FEATURE_NAMES, shap_vals) if f.startswith('Marker')]
                            marker_shaps.sort(key=lambda x: x[1], reverse=True)
                            best_marker, best_val = marker_shaps[0]

                            if best_val > 0:
                                st.markdown(f"""
                                <div class="success-box">
                                ⭐ <strong>Most Feasible Marker for {selected}: {best_marker}</strong><br>
                                This marker <strong>increases yield by +{best_val:.3f} t/ha</strong>
                                and is the best genomic choice for this variety.
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="warn-box">
                                ⚠️ All markers show neutral or negative impact for <strong>{selected}</strong>.
                                Consider different SNP combinations for better yield.
                                </div>
                                """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)

elif not predict_clicked:
    st.markdown("""
    <div class="info-box">
    👆 Upload your lab report CSV and set climate conditions above, then click <strong>Predict Yield</strong> to see results.
    </div>
    """, unsafe_allow_html=True)
