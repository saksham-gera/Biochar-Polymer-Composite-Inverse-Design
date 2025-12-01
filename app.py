import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Page Configuration ---
st.set_page_config(
    page_title="Biochar Composite Predictor",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Biochar-Polymer Composite Inverse Design")
st.markdown("""
This application predicts the required material composition (**Biochar, PANI, MWCNT**) 
based on desired electromagnetic shielding properties.
""")

# # --- Sidebar: Data Upload ---
# st.sidebar.header("1. Upload Data")
# uploaded_file = st.sidebar.file_uploader("Upload 'generated_shielding_data_full.csv'", type=["csv"])

# Define default file path (local fallback)
default_file = 'generated_shielding_data_full.csv'
uploaded_file = None
df = None

# Logic to load data (either from upload or local file)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Uploaded file loaded.")
else:
    try:
        df = pd.read_csv(default_file)
        st.sidebar.info(f"Loaded local file: `{default_file}`")
    except FileNotFoundError:
        st.error("❌ Data not found. Please upload the CSV file in the sidebar.")
        st.stop()

# --- Model Training Section ---
# We use @st.cache_resource so the model doesn't retrain every time you interact with the app
@st.cache_resource
def train_model(data):
    # Features and Targets
    features = ['Frequency_GHz', 'sigma_eff_S_m', 'sigma_star_S_m', 'SE_R_dB', 'SE_A_dB', 'SE_T_dB']
    targets = ['Biochar_wt', 'PANI_wt', 'MWCNT_wt']

    X = data[features]
    y = data[targets]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    
    metrics = {
        'Biochar': {
            'mae': mean_absolute_error(y_test['Biochar_wt'], y_pred[:, 0]),
            'r2': r2_score(y_test['Biochar_wt'], y_pred[:, 0])
        },
        'PANI': {
            'mae': mean_absolute_error(y_test['PANI_wt'], y_pred[:, 1]),
            'r2': r2_score(y_test['PANI_wt'], y_pred[:, 1])
        },
        'MWCNT': {
            'mae': mean_absolute_error(y_test['MWCNT_wt'], y_pred[:, 2]),
            'r2': r2_score(y_test['MWCNT_wt'], y_pred[:, 2])
        }
    }
    
    return model, metrics, features

# Train the model
with st.spinner('Training Random Forest Model...'):
    model, metrics, feature_names = train_model(df)

# --- Display Metrics ---
st.markdown("---")
st.subheader("2. Model Performance (Test Set)")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Biochar Accuracy**")
    st.metric("R² Score", f"{metrics['Biochar']['r2']:.3f}")
    st.metric("MAE (wt%)", f"{metrics['Biochar']['mae']:.3f}")

with col2:
    st.info("**PANI Accuracy**")
    st.metric("R² Score", f"{metrics['PANI']['r2']:.3f}")
    st.metric("MAE (wt%)", f"{metrics['PANI']['mae']:.3f}")

with col3:
    st.info("**MWCNT Accuracy**")
    st.metric("R² Score", f"{metrics['MWCNT']['r2']:.3f}")
    st.metric("MAE (wt%)", f"{metrics['MWCNT']['mae']:.3f}")

# --- Prediction Interface ---
st.markdown("---")
st.subheader("3. Inverse Design: Predict Composition")
st.markdown("Input your desired shielding properties below to get the required material recipe.")

# Create input fields organized in columns
input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    freq = st.number_input("Frequency (GHz)", value=10.0, step=0.1)
    se_r = st.number_input("SE Reflection (dB)", value=18.17, step=0.1)

with input_col2:
    sig_eff = st.number_input("Sigma Effective (S/m)", value=12.86, step=0.1)
    se_a = st.number_input("SE Absorption (dB)", value=5.53, step=0.1)

with input_col3:
    # AUTO-CALCULATED VALUES
    sig_star = sig_eff / 10
    se_t = se_r + se_a

    st.write(f"**Sigma Star (Auto)**: {sig_star:.3f} S/m")
    st.write(f"**SE Total (Auto)**: {se_t:.2f} dB")

# Prediction Button
if st.button("Predict Material Composition", type="primary"):
    # Prepare input array
    input_data = np.array([
        freq,
        sig_eff,
        sig_eff / 10,      # sigma_star
        se_r,
        se_a,
        se_r + se_a        # SE_T
    ]).reshape(1, -1)
    
    # Make Prediction
    prediction = model.predict(input_data)[0]
    
    # Clip values to ensure they aren't negative (physically impossible)
    biochar_pred = max(0, prediction[0])
    pani_pred = max(0, prediction[1])
    mwcnt_pred = max(0, prediction[2])
    
    st.success("Here is your expected composition of materials!")
    
    # Display Result Cards
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.metric("Biochar Content", f"{biochar_pred:.2f} wt%")
    with res_col2:
        st.metric("PANI Content", f"{pani_pred:.2f} wt%")
    with res_col3:
        st.metric("MWCNT Content", f"{mwcnt_pred:.2f} wt%")
        
    # Visual representation of the mix
    chart_data = pd.DataFrame({
        'Material': ['Biochar', 'PANI', 'MWCNT'],
        'Weight %': [biochar_pred, pani_pred, mwcnt_pred]
    })
    st.bar_chart(chart_data, x='Material', y='Weight %')