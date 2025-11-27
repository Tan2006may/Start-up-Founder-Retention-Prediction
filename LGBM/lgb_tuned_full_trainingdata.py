#!/usr/bin/env python3
"""
train_lgbm_full.py
Train final LightGBM model on FULL training data (no validation split)
using best params from tuning and generate submission.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib

# -------------------------
# Preprocessing helpers
# -------------------------
def preprocess_fit(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Numeric: median
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Categorical: mode
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = df[c].astype(str)
        le.fit(df[c])
        df[c] = le.transform(df[c])
        encoders[c] = le

    return df, encoders

def preprocess_apply(df, encoders):
    df = df.copy()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])
        df[c] = df[c].astype(str)
        if c in encoders:
            le = encoders[c]
            mapping = {v: i for i, v in enumerate(le.classes_)}
            default_idx = len(le.classes_)
            df[c] = df[c].map(mapping).fillna(default_idx).astype(int)
        else:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c])
            encoders[c] = le

    return df

# -------------------------
# Main
# -------------------------
def main(args):
    # Load data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

    # Split target and features
    y = train["retention_status"]
    X = train.drop(["retention_status"], axis=1)

    # Preprocess
    X_proc, encs = preprocess_fit(X)
    test_proc = preprocess_apply(test, encs)

    # Encode target
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y)

    # Best params from tuning: baseline_like config
    model = lgb.LGBMClassifier(
        n_estimators=261,          # from best_iter
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        class_weight=None,
        random_state=42,
        n_jobs=-1
    )

    # Train on FULL data
    print("Training LightGBM on FULL training data...")
    model.fit(X_proc, y_enc)

    # Predict on test
    print("Predicting on test data...")
    try:
        test_preds = model.predict(test_proc)
    except Exception:
        test_preds = np.argmax(model.predict_proba(test_proc), axis=1)

    test_labels = target_le.inverse_transform(test_preds)

    # Build submission from sample
    submission = sample.copy()
    # assume last column is the target column in sample_submission
    submission[submission.columns[-1]] = test_labels
    submission.to_csv(args.output, index=False)

    print("Saved final submission to:", args.output)

    # Save model
    joblib.dump(model, "lgbm_full_model.pkl")
    print("Saved trained model to: lgbm_full_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="submission_lgbm_full.csv",
                        help="Output submission filename")
    args = parser.parse_args()
    main(args)
