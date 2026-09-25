import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

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
OUT_FEATURES = "data/processed/adhd_features.csv"
MODEL_PATH = "models/adhd_deploy.pkl"
FEAT_NAMES_PATH = "models/feature_names.pkl"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)


# -------------------------------------------------
# Label Mapping
# -------------------------------------------------
def map_labels(df):

    if "Subtype" in df.columns:
        return df

    if "Class" not in df.columns:
        raise ValueError("Need Class or Subtype column in raw file.")

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

    sub_adhd = np.where(
        score <= q1, 2,
        np.where(score <= q2, 1, 3)
    )

    sub[is_control] = 0
    sub[is_adhd] = sub_adhd[is_adhd]

    df = df.copy()
    df["Subtype"] = sub

    return df


# -------------------------------------------------
# Main Training
# -------------------------------------------------
def main():

    print("Loading raw dataset...")
    df = pd.read_csv(DATA)

    # SPEED IMPROVEMENT
    print("Sampling dataset for faster tuning...")
    df = df.sample(300000, random_state=42)

    if "ID" not in df.columns:
        raise ValueError("Raw dataset must contain an ID column.")

    df = map_labels(df)

    df = reject_bad_rows(df, EEG_COLS, max_abs=1e9)
    df = robust_clip(df, EEG_COLS, z=6.0)

    # -------------------------------------------------
    # Load cached features if available
    # -------------------------------------------------
    if os.path.exists(OUT_FEATURES):

        print("Loading precomputed features...")
        feats = pd.read_csv(OUT_FEATURES)

    else:

        print("Extracting features...")
        feats = extract_tabular_eeg_features(df, EEG_COLS)

        feats["Subtype"] = df["Subtype"].astype(int).values
        feats["ID"] = df["ID"].astype(str).values

        feats.to_csv(OUT_FEATURES, index=False)

        print("Saved features:", OUT_FEATURES)

    X = feats.drop(columns=["Subtype", "ID"])
    y = feats["Subtype"].values
    groups = feats["ID"].values


    # -------------------------------------------------
    # Group split (subject-wise split)
    # -------------------------------------------------
    splitter = GroupShuffleSplit(
        test_size=0.2,
        n_splits=1,
        random_state=42
    )

    tr_idx, te_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]


    # -------------------------------------------------
    # Class weights
    # -------------------------------------------------
    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weight_dict = dict(zip(classes, weights))
    class_weight_dict[0] = class_weight_dict[0] * 2.0

    sample_weights = np.array([
        class_weight_dict[label]
        for label in y_train
    ])


    # -------------------------------------------------
    # Base Model
    # -------------------------------------------------
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42
    )


    # -------------------------------------------------
    # Hyperparameter Search Space
    # -------------------------------------------------
    param_dist = {

        "n_estimators": [200, 400, 600],

        "max_depth": [5, 7, 9],

        "learning_rate": [0.01, 0.02, 0.05],

        "subsample": [0.8, 0.9],

        "colsample_bytree": [0.8, 0.9],

        "min_child_weight": [1, 3],

        "gamma": [0, 0.2],

        "reg_lambda": [1, 2],

        "reg_alpha": [0, 0.5]
    }


    # -------------------------------------------------
    # Randomized Hyperparameter Tuning
    # -------------------------------------------------
    print("Starting Hyperparameter Search...")

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=10,      # reduced from 20
        scoring="accuracy",
        cv=2,           # reduced from 3
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train, sample_weight=sample_weights)

    best_model = search.best_estimator_

    print("\nBest Parameters Found:")
    print(search.best_params_)


    # -------------------------------------------------
    # Final Evaluation
    # -------------------------------------------------
    pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print("\nDeploy Accuracy:", round(acc, 4))
    print(classification_report(y_test, pred))


    # -------------------------------------------------
    # Save Model
    # -------------------------------------------------
    joblib.dump(best_model, MODEL_PATH, compress=3)
    joblib.dump(list(X.columns), FEAT_NAMES_PATH, compress=3)

    print("Saved model to:", MODEL_PATH)


if __name__ == "__main__":
    main()
