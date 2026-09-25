import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing.filters import robust_clip
from src.preprocessing.artifact import reject_bad_rows
from src.features.feature_extraction import extract_tabular_eeg_features


# =========================
# CONFIG
# =========================

MODEL_PATH = "models/adhd_deploy.pkl"
FEAT_NAMES_PATH = "models/feature_names.pkl"

EEG_COLS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4',
    'O1','O2','F7','F8','T7','T8','P7','P8',
    'Fz','Cz','Pz'
]

st.set_page_config(page_title="ADHD EEG Subtype Predictor", layout="wide")

st.title("🧠 ADHD EEG Subtype Prediction")
st.write("Upload EEG CSV file to predict ADHD subtype.")


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEAT_NAMES_PATH)
    return model, feature_names

model, feature_names = load_model()


# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader("Upload EEG CSV File", type=["csv"])

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        if not all(col in df.columns for col in EEG_COLS):
            st.error("Uploaded file does not contain required EEG columns.")
            st.stop()

        st.success("File uploaded successfully!")

        # =========================
        # RAW EEG PREVIEW
        # =========================

        st.subheader("Raw EEG Input (Preview)")
        st.dataframe(df.head(10), use_container_width=True)

        # =========================
        # CLEAN EEG DATA
        # =========================

        with st.spinner("Cleaning EEG signals..."):

            df = reject_bad_rows(df, EEG_COLS, max_abs=1e9)
            df = robust_clip(df, EEG_COLS, z=6.0)

        # =========================
        # FEATURE EXTRACTION
        # =========================

        with st.spinner("Extracting EEG features..."):

            feats = extract_tabular_eeg_features(df, EEG_COLS)

        if feats.shape[0] == 0:
            st.error("Feature extraction produced no data. Please upload a longer EEG recording.")
            st.stop()

        st.success("Feature extraction complete!")

        # =========================
        # ALIGN FEATURES WITH MODEL
        # =========================

        for col in feature_names:
            if col not in feats.columns:
                feats[col] = 0

        X = feats[feature_names]

        # =========================
        # PREDICTION
        # =========================

        probs = model.predict_proba(X)
        preds = np.argmax(probs, axis=1)

        avg_prob = probs.mean(axis=0)
        final_pred = np.argmax(avg_prob)
        final_conf = np.max(avg_prob)

        subtype_map = {
            0: "Control",
            1: "ADHD - Inattentive",
            2: "ADHD - Hyperactive",
            3: "ADHD - Combined"
        }

        # =========================
        # FINAL RESULT
        # =========================

        st.subheader("Final ADHD Subtype Prediction")

        st.success(f"Predicted Subtype: {subtype_map[final_pred]}")
        st.info(f"Confidence: {final_conf:.4f}")

        # =========================
        # SEGMENT DISTRIBUTION
        # =========================

        st.subheader("Segment-wise Prediction Distribution")

        dist_df = pd.DataFrame({
            "Subtype": list(subtype_map.values()),
            "Probability": avg_prob
        })

        st.dataframe(dist_df, use_container_width=True)
        st.bar_chart(dist_df.set_index("Subtype"))

        # =========================
        # BRAIN REGION ANALYSIS
        # =========================

        st.subheader("Regional Brain Activity Analysis")

        regions = {
            "Frontal": ['Fp1','Fp2','F3','F4','F7','F8','Fz'],
            "Central": ['C3','C4','Cz'],
            "Parietal": ['P3','P4','P7','P8','Pz'],
            "Occipital": ['O1','O2'],
            "Temporal": ['T7','T8']
        }

        region_activity = {}

        for region, channels in regions.items():
            region_activity[region] = df[channels].mean().mean()

        region_df = pd.DataFrame(region_activity.items(),
                                 columns=["Region", "Mean Activity"])

        st.dataframe(region_df, use_container_width=True)
        st.bar_chart(region_df.set_index("Region"))

        # =========================
        # Z-SCORE HEATMAP
        # =========================

        st.subheader("EEG Channel Activity Heatmap")

        z_scores = (df[EEG_COLS] - df[EEG_COLS].mean()) / df[EEG_COLS].std()
        z_mean = z_scores.mean()

        fig, ax = plt.subplots(figsize=(18,3))

        sns.heatmap(
            z_mean.values.reshape(1,-1),
            annot=True,
            cmap="coolwarm",
            xticklabels=EEG_COLS,
            yticklabels=["Z-score"],
            ax=ax
        )

        st.pyplot(fig)

        # =========================
        # SEGMENT PREDICTIONS
        # =========================

        st.subheader("Segment-wise Predictions")

        results = pd.DataFrame({
            "Predicted Subtype": [subtype_map[p] for p in preds],
            "Confidence": np.max(probs, axis=1)
        })

        st.dataframe(results.head(10), use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
