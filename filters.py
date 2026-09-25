import numpy as np
import pandas as pd

def robust_clip(df: pd.DataFrame, cols, z=6.0) -> pd.DataFrame:
    """
    Simple robust artifact suppression: clips extreme values per channel.
    Works for tabular/segment-level EEG values.
    """
    out = df.copy()
    x = out[cols].astype(float).to_numpy()
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0) + 1e-8
    lo = mu - z * sd
    hi = mu + z * sd
    x = np.clip(x, lo, hi)
    out[cols] = x
    return out
