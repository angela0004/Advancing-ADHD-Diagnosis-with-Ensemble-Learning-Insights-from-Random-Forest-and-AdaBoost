import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis


# ===============================
# BAND POWER FUNCTION
# ===============================
def bandpower(signal, fs, band):
    freqs, psd = welch(signal, fs)

    mask = (freqs >= band[0]) & (freqs <= band[1])

    return np.trapz(psd[mask], freqs[mask])


# ===============================
# FEATURE EXTRACTION FUNCTION
# ===============================
def extract_tabular_eeg_features(df: pd.DataFrame, eeg_cols=None, fs=128):

    if eeg_cols is None:
        eeg_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # convert to numpy for faster computation
    X = df[eeg_cols].values

    features_df = pd.DataFrame(X, columns=eeg_cols)

    # -----------------------------
    # FAST STATISTICAL FEATURES
    # -----------------------------
    features_df["row_mean"] = np.mean(X, axis=1)
    features_df["row_std"] = np.std(X, axis=1)
    features_df["row_min"] = np.min(X, axis=1)
    features_df["row_max"] = np.max(X, axis=1)
    features_df["row_median"] = np.median(X, axis=1)

    # faster skew/kurtosis
    features_df["row_skew"] = skew(X, axis=1)
    features_df["row_kurtosis"] = kurtosis(X, axis=1)

    # -----------------------------
    # ENERGY
    # -----------------------------
    features_df["row_energy"] = np.sum(X ** 2, axis=1)

    # -----------------------------
    # ZERO CROSSING RATE
    # -----------------------------
    features_df["row_zcr"] = np.sum(np.diff(np.sign(X), axis=1) != 0, axis=1)

    # -----------------------------
    # EEG BAND POWERS (FAST)
    # -----------------------------
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
    }

    for i, col in enumerate(eeg_cols):

        signal = X[:, i]

        for band_name, band_range in bands.items():

            power = bandpower(signal, fs, band_range)

            features_df[f"{col}_{band_name}_power"] = power

    # -----------------------------
    # THETA / BETA RATIO
    # -----------------------------
    for col in eeg_cols:

        theta = features_df[f"{col}_theta_power"]
        beta = features_df[f"{col}_beta_power"]

        features_df[f"{col}_theta_beta_ratio"] = theta / (beta + 1e-6)

    # -----------------------------
    # CLEAN VALUES
    # -----------------------------
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(0, inplace=True)

    return features_df


# ===============================
# LOAD EEG DATASET
# ===============================
print("Loading dataset...")

df = pd.read_csv("data/raw/shahed_eeg.csv")

print("Dataset shape:", df.shape)

# ===============================
# CALL FEATURE EXTRACTION
# ===============================
print("Extracting features...")

features_df = extract_tabular_eeg_features(df)

# ===============================
# PRINT FEATURES
# ===============================
print("\nFirst 5 rows of extracted features:")
print(features_df.head())

print("\nSingle feature row:")
print(features_df.iloc[0])

print("\nTotal features generated:", features_df.shape[1])
