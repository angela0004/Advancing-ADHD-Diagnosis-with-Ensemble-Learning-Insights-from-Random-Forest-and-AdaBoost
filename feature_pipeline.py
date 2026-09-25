import numpy as np
import pandas as pd

from src.preprocessing.filters import bandpass_filter, notch_filter
from src.preprocessing.windowing import make_windows
from src.preprocessing.artifact import reject_artifacts
from src.features.time_features import extract_time_features
from src.features.freq_features import extract_freq_features
from src.features.connectivity import extract_connectivity_features

def featurize_timeseries(df: pd.DataFrame, channels, fs, bandpass, notch_hz, win, step):
    """
    df: dataframe containing only channel columns in time order
    returns: features_df (N_windows, n_features)
    """
    x = df[channels].to_numpy(dtype=float)     # (T, C)
    x = notch_filter(x, fs, notch_hz)
    x = bandpass_filter(x, fs, bandpass[0], bandpass[1])

    windows = make_windows(x, win=win, step=step)     # (N, win, C)
    windows = reject_artifacts(windows)

    rows = []
    for w in windows:
        feats = {}
        feats.update(extract_time_features(w, channels))
        feats.update(extract_freq_features(w, fs, channels))
        feats.update(extract_connectivity_features(w, fs, channels))
        rows.append(feats)

    return pd.DataFrame(rows)
