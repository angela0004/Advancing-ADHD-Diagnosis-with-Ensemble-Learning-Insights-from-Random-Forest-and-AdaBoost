import numpy as np
import pandas as pd

def persona_normalize(df: pd.DataFrame, cols, eps=1e-8) -> pd.DataFrame:
    """
    PERSONA: subject-level normalization (within the uploaded file / subject batch).
    Normalizes each channel using mean/std computed from that subject's data.
    """
    out = df.copy()
    x = out[cols].astype(float).to_numpy()
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0) + eps
    out[cols] = (x - mu) / sd
    return out
