# app.py
"""
Streamlit prediction app for Supply Chain Demand Forecasting.

Expecting these files in the same directory (created by your training script):
 - ensemble_model.pkl
 - rf_model.pkl
 - scaler2.pkl
 - label_encoders.pkl
 - feature_columns.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.exceptions import NotFittedError

st.set_page_config(layout="wide", page_title="Supply Chain - Demand Predictor")

@st.cache_resource
def load_artifacts():
    artifacts = {}
    artifacts['ensemble'] = joblib.load("E:/Unified_Mentor_Data_Analyst_Internship_Projects/Supply chain Management Project/ensemble_model.pkl")
    artifacts['rf'] = joblib.load("E:/Unified_Mentor_Data_Analyst_Internship_Projects/Supply chain Management Project/rf_model.pkl")
    artifacts['scaler'] = joblib.load("E:/Unified_Mentor_Data_Analyst_Internship_Projects/Supply chain Management Project/scaler2.pkl")
    artifacts['label_encoders'] = joblib.load("E:/Unified_Mentor_Data_Analyst_Internship_Projects/Supply chain Management Project/label_encoders.pkl")
    artifacts['feature_columns'] = joblib.load("E:/Unified_Mentor_Data_Analyst_Internship_Projects/Supply chain Management Project/feature_columns.pkl")
    return artifacts

art = load_artifacts()
ensemble_model = art['ensemble']
rf_model = art['rf']
scaler = art['scaler']
label_encoders = art['label_encoders']  # dict: original_col_name -> LabelEncoder
feature_columns = art['feature_columns']  # list of expected feature column names

st.title("📦 Supply Chain Demand Forecasting App")
st.markdown("Use this app to predict `Number of products sold` for new records (single or batch).")

# Sidebar: choose single or batch
mode = st.sidebar.radio("Mode", ["Single record (form)", "Batch upload (CSV)"])

def prepare_input_df(raw_df: pd.DataFrame):
    """
    Align raw_df to feature_columns:
     - Ensure all expected cols exist (fill missing with 0 or median)
     - Apply label encoders for categorical columns saved in label_encoders dict
     - Convert to numeric and scale with saved scaler
    Returns: scaled ndarray, prepared_df
    """
    dfp = raw_df.copy()
    # Fill missing columns
    for col in feature_columns:
        if col not in dfp.columns:
            # crude fill: 0
            dfp[col] = 0

    # If encoders are present for original categorical column names,
    # we expect encoded column names in feature_columns or original columns.
    # We'll attempt to transform columns that exist both in label_encoders keys and dfp.
    for orig_col, le in label_encoders.items():
        # two possibilities in features: either f"{orig_col}_encoded" exists or orig_col exists
        enc_col = f"{orig_col}_encoded"
        if enc_col in feature_columns:
            # if raw df contains orig_col values, encode them and write to enc_col
            if orig_col in dfp.columns:
                # handle unseen labels gracefully: map unseen -> -1 then to a numeric label
                try:
                    dfp[enc_col] = le.transform(dfp[orig_col].astype(str))
                except Exception:
                    # fallback: map classes to indices, unseen -> -1
                    mapping = {c: i for i, c in enumerate(le.classes_)}
                    dfp[enc_col] = dfp[orig_col].astype(str).map(mapping).fillna(-1).astype(int)
            elif enc_col in dfp.columns:
                # column already provided as numeric, nothing to do
                dfp[enc_col] = pd.to_numeric(dfp[enc_col], errors='coerce').fillna(0)
            else:
                # create default
                dfp[enc_col] = 0

    # Ensure numeric types for features
    X = dfp[feature_columns].copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale
    X_scaled = scaler.transform(X)
    return X_scaled, X

def predict_with_interval(rf_model, X_scaled, percentile=90):
    """
    Compute RF-based prediction intervals (per-sample).
    rf_model should be a fitted RandomForestRegressor.
    Returns: median_pred, lower, upper
    """
    try:
        # Each estimator predicts on the scaled X -> shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X_scaled) for tree in rf_model.estimators_])
        lower = np.percentile(all_preds, (100 - percentile) / 2, axis=0)
        upper = np.percentile(all_preds, 100 - (100 - percentile) / 2, axis=0)
        median = np.median(all_preds, axis=0)
        return median, lower, upper
    except Exception as e:
        # if rf_model doesn't have estimators_ (e.g., not RF) fallback to point preds
        pred = rf_model.predict(X_scaled)
        return pred, pred, pred

if mode == "Single record (form)":
    st.subheader("Enter features for a single record")
    # Build a friendly form for common numeric features and categorical picks if available.
    # To keep form reasonable, we show top numeric columns and categorical keys (limit to 8).
    numeric_preview = [c for c in feature_columns if "_encoded" not in c][:12]
    cat_preview = list(label_encoders.keys())[:6]

    with st.form("single_form"):
        cols = st.columns(2)
        inputs = {}
        # numeric inputs (first half)
        for i, col in enumerate(numeric_preview):
            with cols[i % 2]:
                inputs[col] = st.text_input(col, value="0")

        # categorical selects
        st.markdown("**Categorical fields**")
        for orig_col in cat_preview:
            classes = label_encoders.get(orig_col).classes_.tolist()
            # add an option for unseen
            classes = ["__UNSEEN__"] + classes
            val = st.selectbox(f"{orig_col}", options=classes, index=0)
            inputs[orig_col] = val

        submitted = st.form_submit_button("Predict")
        if submitted:
            raw_df = pd.DataFrame([inputs])
            X_scaled, X_prepared = prepare_input_df(raw_df)
            # ensemble prediction
            ens_pred = ensemble_model.predict(X_scaled)[0]
            rf_pred = rf_model.predict(X_scaled)[0]
            median, lower, upper = predict_with_interval(rf_model, X_scaled, percentile=90)
            st.success(f"Ensemble prediction: {ens_pred:.2f} units")
            st.info(f"Random Forest prediction: {rf_pred:.2f} units")
            st.write(f"90% interval (RF): {lower[0]:.0f} — {upper[0]:.0f} units (median {median[0]:.0f})")
            # show prepared features
            st.markdown("**Prepared features used for prediction:**")
            st.dataframe(X_prepared.T)

else:
    st.subheader("Batch prediction (CSV upload)")
    st.markdown("Upload a CSV with columns that include your original dataset columns (e.g., Product Type, Price, etc.). App will align features automatically.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df_in)} rows.")
        st.dataframe(df_in.head(5))
        if st.button("Run batch prediction"):
            X_scaled, X_prepared = prepare_input_df(df_in)
            ens_preds = ensemble_model.predict(X_scaled)
            rf_preds = rf_model.predict(X_scaled)
            median, lower, upper = predict_with_interval(rf_model, X_scaled, percentile=90)
            out = df_in.copy().reset_index(drop=True)
            out['pred_ensemble'] = ens_preds
            out['pred_rf'] = rf_preds
            out['pred_rf_lower_90'] = lower
            out['pred_rf_median'] = median
            out['pred_rf_upper_90'] = upper
            st.write(out.head(10))
            # allow download
            csv = out.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.write("Model loaded. Runtimes are instantaneous for small inputs.")
st.success("✅ Predictions completed! Download your results below.")

