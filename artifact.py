import numpy as np
import pandas as pd

def reject_bad_rows(df: pd.DataFrame, cols, max_abs=1e6) -> pd.DataFrame:
    """
    Removes rows with impossible amplitudes.
    Adjust max_abs based on your dataset scale.
    """
    x = df[cols].astype(float)
    mask = (x.abs() < max_abs).all(axis=1)
    return df.loc[mask].reset_index(drop=True)
