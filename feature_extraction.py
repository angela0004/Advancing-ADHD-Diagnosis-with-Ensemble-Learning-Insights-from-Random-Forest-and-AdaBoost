import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch


# ===============================
# BAND POWER FUNCTION
# ===============================
def bandpower(signal, fs, band):
    freqs, psd = welch(signal, fs)

    mask = (freqs >= band[0]) & (freqs <= band[1])

    return np.trapz(psd[mask], freqs[mask])


# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_tabular_eeg_features(df: pd.DataFrame, eeg_cols=None, fs=128) -> pd.DataFrame:
    """
    Advanced EEG feature engineering.

    Includes:
    - Statistical features
    - Signal energy
    - Zero crossing rate
    - EEG band powers (window based)
    - Theta/Beta ratio
    """

    if eeg_cols is None:
        eeg_cols = df.select_dtypes(include=["number"]).columns.tolist()

    features_df = df[eeg_cols].copy()

    # ===============================
    # STATISTICAL FEATURES
    # ===============================

    features_df["row_mean"] = features_df.mean(axis=1)
    features_df["row_std"] = features_df.std(axis=1)
    features_df["row_min"] = features_df.min(axis=1)
    features_df["row_max"] = features_df.max(axis=1)
    features_df["row_median"] = features_df.median(axis=1)

    # Skewness and kurtosis
    features_df["row_skew"] = features_df.apply(skew, axis=1)
    features_df["row_kurtosis"] = features_df.apply(kurtosis, axis=1)

    # ===============================
    # ENERGY
    # ===============================

    features_df["row_energy"] = (features_df ** 2).sum(axis=1)

    # ===============================
    # ZERO CROSSING RATE
    # ===============================

    def zero_cross(row):
        return np.sum(np.diff(np.sign(row)) != 0)

    features_df["row_zcr"] = features_df.apply(zero_cross, axis=1)

    # ===============================
    # EEG BAND POWERS (WINDOW BASED)
    # ===============================

    window_size = fs  # 1 second window

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
    }

    for col in eeg_cols:

        for band_name, band_range in bands.items():

            features_df[f"{col}_{band_name}_power"] = (
                df[col]
                .rolling(window=window_size)
                .apply(lambda x: bandpower(x, fs, band_range), raw=True)
            )

    # ===============================
    # THETA / BETA RATIO
    # ===============================

    for col in eeg_cols:

        theta_col = f"{col}_theta_power"
        beta_col = f"{col}_beta_power"

        features_df[f"{col}_theta_beta_ratio"] = (
            features_df[theta_col] / (features_df[beta_col] + 1e-6)
        )

    # ===============================
    # CLEAN OUTPUT
    # ===============================

    features_df = features_df.replace([np.inf, -np.inf], np.nan)

    # rolling windows create NaNs at start
    features_df = features_df.dropna().reset_index(drop=True)

    # ===============================
    # DEBUG PRINTS
    # ===============================

    print("\nExtracted Feature Table (first 5 rows):")
    print(features_df.head())

    print("\nSingle compressed feature row:")
    print(features_df.iloc[0])

    print("\nTotal Features Generated:", features_df.shape[1])

    return features_df
