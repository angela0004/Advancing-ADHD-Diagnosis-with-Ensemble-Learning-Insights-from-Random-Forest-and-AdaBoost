import numpy as np
import pandas as pd

# EEG channels used in your model
channels = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4',
    'O1','O2','F7','F8','T7','T8','P7','P8',
    'Fz','Cz','Pz'
]

# generate 500 rows of realistic EEG values
rows = 500

data = np.random.normal(loc=100, scale=200, size=(rows, len(channels)))

df = pd.DataFrame(data, columns=channels)

df.to_csv("demo_test_eeg.csv", index=False)

print("Test EEG file created: demo_test_eeg.csv")
