#!/usr/bin/env python3
"""
train_models.py -- Windows-friendly, robust version
Usage examples:
  python train_models.py --model lgb --output submission_lgb.csv
  python train_models.py --model xgb --output submission_xgb.csv
  python train_models.py --model rf  --output submission_rf.csv
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# -------------------------
# Preprocessing utilities
# -------------------------
def preprocess_fit(df):
    """Fit preprocessing on df (train). Returns processed df and encoders dict."""
    df = df.copy()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Fill numeric with median
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Fill categorical with mode
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    # Fit LabelEncoders
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        # Convert to str to avoid issues with mixed types / NaN after fill
        df[c] = df[c].astype(str)
        le.fit(df[c])
        df[c] = le.transform(df[c])
        encoders[c] = le

    return df, encoders

def preprocess_apply(df, encoders):
    """Apply fitted encoders to df (test). Unseen labels mapped to new index (= len(classes))."""
    df = df.copy()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Fill numeric with median of df (fallback)
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Fill categorical with mode (fallback)
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    # Apply encoders
    for c in cat_cols:
        df[c] = df[c].astype(str)
        if c in encoders:
            le = encoders[c]
            mapping = {v: i for i, v in enumerate(le.classes_)}
            default_index = len(le.classes_)  # unseen -> this new index
            df[c] = df[c].map(mapping).fillna(default_index).astype(int)
        else:
            # If encoder not available, fit a local encoder (rare)
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c])
            encoders[c] = le

    return df

# -------------------------
# Main
# -------------------------
def main(args):
    # Load CSVs (expect them next to this script)
    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

    # Split features and target
    y = train["retention_status"]
    X = train.drop(["retention_status"], axis=1)

    # Preprocess train (fit encoders) and test (apply)
    X_proc, encs = preprocess_fit(X)
    test_proc = preprocess_apply(test, encs)

    # Encode target
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_proc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model_name = args.model.lower()

    # -------------------------
    # LightGBM (with callbacks - compatible across versions)
    # -------------------------
    if model_name in ("lgb", "lightgbm"):
        try:
            import lightgbm as lgb
            from lightgbm import early_stopping, log_evaluation
        except Exception as e:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm") from e

        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=50)]
        )

    # -------------------------
    # XGBoost
    # -------------------------
    elif model_name in ("xgb", "xgboost"):
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise ImportError("xgboost not installed. Run: pip install xgboost") from e

        model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            tree_method="hist",
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=50
        )

    # -------------------------
    # Random Forest
    # -------------------------
    elif model_name in ("rf", "randomforest"):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

    else:
        raise ValueError("Unsupported model. Choose from: lgb | xgb | rf")

    # -------------------------
    # Evaluation on validation set
    # -------------------------
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f"\nValidation Accuracy: {acc:.4f}")
    print(f"Validation F1:       {f1:.4f}\n")
    print("Classification report:")
    print(classification_report(y_val, y_pred, target_names=target_le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred))

    # -------------------------
    # Predict on test and save submission
    # -------------------------
    try:
        test_preds = model.predict(test_proc)
    except Exception:
        # fallback to predict_proba
        test_preds = np.argmax(model.predict_proba(test_proc), axis=1)

    test_labels = target_le.inverse_transform(test_preds)

    submission = sample.copy()
    # put predictions into last column of sample submission
    submission[submission.columns[-1]] = test_labels
    submission.to_csv(args.output, index=False)
    print("\nSaved submission to:", args.output)

    # save model for later reuse
    joblib.dump(model, f"model_{model_name}.pkl")
    print("Saved model to:", f"model_{model_name}.pkl")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="lgb", help="lgb | xgb | rf")
    p.add_argument("--output", type=str, default="submission.csv")
    args = p.parse_args()
    main(args)
