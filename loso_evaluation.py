import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.preprocessing.filters import robust_clip
from src.preprocessing.artifact import reject_bad_rows
from src.features.feature_extraction import extract_tabular_eeg_features


EEG_COLS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4',
    'O1','O2','F7','F8','T7','T8','P7','P8',
    'Fz','Cz','Pz'
]

DATA = "data/raw/shahed_eeg.csv"
CHECKPOINT_FILE = "loso_progress.csv"


def map_labels(df):
    if "Subtype" in df.columns:
        return df

    if "Class" not in df.columns:
        raise ValueError("Need Class or Subtype column.")

    s = df["Class"].astype(str).str.lower()
    is_control = s.str.contains("control|td|healthy|normal|0")
    is_adhd = ~is_control

    X = df[EEG_COLS].astype(float)

    score = (
        X.var(axis=1) +
        (X[["F3","F4","Fz"]].mean(axis=1) -
         X[["P3","P4","Pz"]].mean(axis=1)).abs()
    )

    q1, q2 = np.quantile(score[is_adhd], [0.33, 0.66])

    sub = np.zeros(len(df), dtype=int)
    sub_adhd = np.where(score <= q1, 2,
                        np.where(score <= q2, 1, 3))

    sub[is_control] = 0
    sub[is_adhd] = sub_adhd[is_adhd]

    df = df.copy()
    df["Subtype"] = sub
    return df


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA)

    if "ID" not in df.columns:
        raise ValueError("Dataset must contain ID column")

    df = map_labels(df)

    df = reject_bad_rows(df, EEG_COLS, max_abs=1e9)
    df = robust_clip(df, EEG_COLS, z=6.0)

    print("Extracting features...")
    feats = extract_tabular_eeg_features(df, EEG_COLS)

    feats["Subtype"] = df["Subtype"].astype(int).values
    feats["ID"] = df["ID"].astype(str).values

    X = feats.drop(columns=["Subtype", "ID"])
    y = feats["Subtype"].values
    groups = feats["ID"].values

    unique_ids = np.unique(groups)

    # =========================
    # LOAD CHECKPOINT
    # =========================
    if os.path.exists(CHECKPOINT_FILE):
        progress_df = pd.read_csv(CHECKPOINT_FILE)
        done_ids = set(progress_df["ID"].astype(str))
        all_preds = progress_df["pred"].tolist()
        all_true = progress_df["true"].tolist()
        print(f"Resuming... Completed {len(done_ids)} subjects")
    else:
        done_ids = set()
        all_preds = []
        all_true = []

    print(f"Total Subjects: {len(unique_ids)}")

    # =========================
    # LOSO LOOP
    # =========================
    for i, test_id in enumerate(unique_ids):

        if str(test_id) in done_ids:
            continue

        print(f"\nLOSO → Testing on ID: {test_id}")

        train_idx = groups != test_id
        test_idx = groups == test_id

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = XGBClassifier(
            n_estimators=200,   # 🔥 reduced for speed
            max_depth=7,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist"
        )

        clf.fit(X_train_s, y_train)

        preds = clf.predict(X_test_s)

        final_pred = np.bincount(preds).argmax()
        true_label = np.bincount(y_test).argmax()

        all_preds.append(final_pred)
        all_true.append(true_label)

        done_ids.add(str(test_id))

        print(f"Predicted: {final_pred} | Actual: {true_label}")

        # =========================
        # SAVE PROGRESS
        # =========================
        pd.DataFrame({
            "ID": list(done_ids),
            "pred": all_preds,
            "true": all_true
        }).to_csv(CHECKPOINT_FILE, index=False)

    print("\n========== FINAL RESULTS ==========")

    acc = accuracy_score(all_true, all_preds)
    print("LOSO Accuracy:", round(acc, 4))

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds))


if __name__ == "__main__":
    main()
