import numpy as np
import pandas as pd

def phys_aug_tabular(X: np.ndarray, noise_std=0.02, scale_jitter=0.02, seed=42) -> np.ndarray:
    """
    Physiologically reasonable augmentation for feature vectors:
    - small gaussian noise
    - small multiplicative scaling
    """
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_std, size=X.shape)
    scale = 1.0 + rng.normal(0, scale_jitter, size=(X.shape[0], 1))
    return (X * scale) + noise
