import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from src.preprocessing.filters import robust_clip
from src.preprocessing.artifact import reject_bad_rows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EEG_COLS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4',
    'O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz'
]

DATA = "data/raw/shahed_eeg.csv"
MODEL_PATH = "models/cnn_model.pt"

WINDOW = 256
STRIDE = 128


# ===========================
# CNN MODEL
# ===========================

class CNN1D(nn.Module):
    def __init__(self, num_classes=2):   # 🔥 changed to 2 classes
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(19, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)


# ===========================
# WINDOW CREATION
# ===========================

def create_windows(df, labels):
    X = df[EEG_COLS].values
    y = labels

    windows = []
    targets = []

    for i in range(0, len(X) - WINDOW, STRIDE):
        windows.append(X[i:i+WINDOW].T)
        targets.append(y[i])

    return np.array(windows), np.array(targets)


# ===========================
# MAIN TRAINING FUNCTION
# ===========================

def main():
    df = pd.read_csv(DATA)

    # Clean EEG data
    df = reject_bad_rows(df, EEG_COLS, max_abs=1e9)
    df = robust_clip(df, EEG_COLS, z=6.0)

    # ===============================
    # FIX LABEL COLUMN
    # ===============================

    print("Unique Classes:", df["Class"].unique())
    print("Class Distribution:")
    print(df["Class"].value_counts())

    # Convert string labels → numeric
    label_map = {
        "Control": 0,
        "ADHD": 1
    }

    df["Class"] = df["Class"].map(label_map)

    # Drop rows if mapping failed
    df = df.dropna(subset=["Class"])

    y = df["Class"].values.astype(int)
    groups = df["ID"].values

    # ===============================
    # SUBJECT-WISE SPLIT
    # ===============================

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    tr_idx, te_idx = next(splitter.split(df, y, groups=groups))

    df_train, df_test = df.iloc[tr_idx], df.iloc[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    X_train, y_train = create_windows(df_train, y_train)
    X_test, y_test = create_windows(df_test, y_test)

    print("Train windows:", X_train.shape)
    print("Test windows:", X_test.shape)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    # ===============================
    # MODEL
    # ===============================

    model = CNN1D(num_classes=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # ===============================
    # TRAINING LOOP
    # ===============================

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(1)
            acc = (preds == y_test).float().mean().item()

        print(f"Epoch {epoch+1:02d} | Loss {loss.item():.4f} | Acc {acc:.4f}")

    # ===============================
    # SAVE MODEL
    # ===============================

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print("✅ CNN model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
