#!/usr/bin/env python3
"""
train_xgb_fe_te_kfold.py

XGBoost + Feature Engineering + K-Fold Target Encoding + Stratified K-Fold CV.

- Uses same general FE/TE ideas as your CatBoost best model.
- Prints OOF (out-of-fold) Accuracy & F1.
- Produces a Kaggle-ready submission CSV.

Usage:
    python train_xgb_fe_te_kfold.py --output submission_xgb_fe_te_kfold.csv
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


# -------------------------
# Feature Engineering
# -------------------------
def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_cols = [
        "founder_age",
        "years_with_startup",
        "monthly_revenue_generated",
        "funding_rounds_led",
        "distance_from_investor_hub",
        "num_dependents",
        "years_since_founding",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    # Base ratios
    if "years_with_startup" in df.columns and "founder_age" in df.columns:
        df["experience_ratio"] = df["years_with_startup"] / (df["founder_age"] + 1)

    if "funding_rounds_led" in df.columns and "years_since_founding" in df.columns:
        df["funding_per_year"] = df["funding_rounds_led"] / (df["years_since_founding"] + 1)

    if "monthly_revenue_generated" in df.columns and "years_since_founding" in df.columns:
        df["revenue_per_year"] = df["monthly_revenue_generated"] / (df["years_since_founding"] + 1)

    if "num_dependents" in df.columns and "founder_age" in df.columns:
        df["dependents_ratio"] = df["num_dependents"] / (df["founder_age"] + 1)

    if "distance_from_investor_hub" in df.columns and "years_since_founding" in df.columns:
        df["distance_adjusted"] = df["distance_from_investor_hub"] / (df["years_since_founding"] + 1)

    # Ordinal-like mappings for rating-type cols
    def map_rating(series, mapping):
        return series.astype(str).str.strip().str.lower().map(mapping).astype(float)

    rating_map = {
        "poor": 1,
        "average": 2,
        "medium": 2,
        "good": 3,
        "excellent": 4,
    }
    sat_map = {
        "low": 1,
        "medium": 2,
        "neutral": 2,
        "high": 3,
        "satisfied": 3,
    }
    rep_map = {
        "poor": 1,
        "fair": 2,
        "moderate": 2,
        "good": 3,
        "excellent": 4,
        "high": 4,
        "low": 1,
    }
    vis_map = {"low": 1, "medium": 2, "high": 3}

    if "work_life_balance_rating" in df.columns:
        df["wlb_score"] = map_rating(df["work_life_balance_rating"], rating_map)
    if "venture_satisfaction" in df.columns:
        df["satisfaction_score"] = map_rating(df["venture_satisfaction"], sat_map)
    if "startup_performance_rating" in df.columns:
        df["performance_score"] = map_rating(df["startup_performance_rating"], rating_map)
    if "startup_reputation" in df.columns:
        df["reputation_score"] = map_rating(df["startup_reputation"], rep_map)
    if "founder_visibility" in df.columns:
        df["visibility_score"] = map_rating(df["founder_visibility"], vis_map)

    # Binning (optional, but can help)
    if "founder_age" in df.columns:
        df["founder_age_bin"] = pd.cut(
            df["founder_age"],
            bins=[0, 25, 35, 45, 60, 100],
            labels=["very_young", "young", "mid", "senior", "old"],
            include_lowest=True,
        )

    if "monthly_revenue_generated" in df.columns:
        try:
            df["revenue_bin"] = pd.qcut(df["monthly_revenue_generated"], q=5, duplicates="drop")
        except Exception:
            df["revenue_bin"] = df["monthly_revenue_generated"]

    if "distance_from_investor_hub" in df.columns:
        try:
            df["distance_bin"] = pd.qcut(df["distance_from_investor_hub"], q=5, duplicates="drop")
        except Exception:
            df["distance_bin"] = df["distance_from_investor_hub"]

    if "years_since_founding" in df.columns:
        df["years_since_founding_bin"] = pd.cut(
            df["years_since_founding"],
            bins=[-1, 1, 3, 5, 10, 100],
            labels=["new", "early", "growing", "mature", "old"],
            include_lowest=True,
        )

    # Drop ID column
    if "founder_id" in df.columns:
        df = df.drop(columns=["founder_id"])

    return df


# -------------------------
# K-Fold Target Encoding
# -------------------------
def kfold_target_encode(train_df, test_df, y, cols, n_splits=5, smoothing=10):
    """
    Leakage-safe K-Fold target encoding.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    y = np.array(y)
    global_mean = y.mean()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in cols:
        if col not in train_df.columns:
            continue

        train_encoded = np.zeros(len(train_df))
        test_fold_encodings = []

        for tr_idx, val_idx in skf.split(train_df, y):
            X_tr_col = train_df.iloc[tr_idx][col]
            y_tr = y[tr_idx]

            temp = pd.DataFrame({col: X_tr_col, "target": y_tr})
            stats = temp.groupby(col)["target"].agg(["mean", "count"])
            stats["smooth"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
                stats["count"] + smoothing
            )
            enc_map = stats["smooth"]

            train_encoded[val_idx] = train_df.iloc[val_idx][col].map(enc_map).fillna(global_mean).values
            test_fold_encodings.append(test_df[col].map(enc_map).fillna(global_mean).values)

        test_encoded = np.mean(test_fold_encodings, axis=0)

        train_df[col + "_te"] = train_encoded
        test_df[col + "_te"] = test_encoded

    return train_df, test_df


# -------------------------
# Main
# -------------------------
def main(args):
    # Load data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

    target_col = "retention_status"
    y = train[target_col]
    X = train.drop(columns=[target_col])

    # Encode target
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y)

    # Feature engineering
    X_fe = add_feature_engineering(X)
    test_fe = add_feature_engineering(test)

    # K-Fold target encoding on selected categoricals
    te_cols = ["founder_role", "education_background", "personal_status", "startup_stage", "team_size_category"]

    X_fe_te, test_fe_te = kfold_target_encode(X_fe, test_fe, y_enc, te_cols, n_splits=5, smoothing=10)

    # Handle remaining categoricals via LabelEncoder
    X_all = X_fe_te.copy()
    test_all = test_fe_te.copy()

    cat_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()

    for c in cat_cols:
        # Fit on train + test combined (just to include all categories, no target leakage here)
        combined = pd.concat([X_all[c], test_all[c]], axis=0).astype(str).fillna("Unknown")
        le_c = LabelEncoder()
        le_c.fit(combined)

        X_all[c] = le_c.transform(X_all[c].astype(str).fillna("Unknown"))
        test_all[c] = le_c.transform(test_all[c].astype(str).fillna("Unknown"))

    # Final matrices
    X_values = X_all.reset_index(drop=True)
    test_values = test_all.reset_index(drop=True)
    y_values = y_enc

    # XGBoost model config
    def make_xgb():
        return XGBClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

    # Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X_values))
    oof_preds = np.zeros(len(X_values), dtype=int)
    test_proba_folds = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_values, y_values), 1):
        print(f"\n===== XGBoost Fold {fold} =====")

        X_tr = X_values.iloc[tr_idx]
        y_tr = y_values[tr_idx]
        X_val = X_values.iloc[val_idx]
        y_val = y_values[val_idx]

        model = make_xgb()

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=100,
            early_stopping_rounds=100,
        )

        best_iter = model.best_iteration if hasattr(model, "best_iteration") else None
        if best_iter is not None:
            val_proba = model.predict_proba(X_val, iteration_range=(0, best_iter + 1))[:, 1]
            test_proba = model.predict_proba(test_values, iteration_range=(0, best_iter + 1))[:, 1]
        else:
            val_proba = model.predict_proba(X_val)[:, 1]
            test_proba = model.predict_proba(test_values)[:, 1]

        oof_proba[val_idx] = val_proba
        fold_preds = (val_proba >= 0.5).astype(int)
        oof_preds[val_idx] = fold_preds

        test_proba_folds.append(test_proba)

        fold_acc = accuracy_score(y_val, fold_preds)
        fold_f1 = f1_score(y_val, fold_preds)
        print(f"Fold {fold} Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}")

    # OOF metrics
    oof_acc = accuracy_score(y_values, oof_preds)
    oof_f1 = f1_score(y_values, oof_preds)

    print("\n==== Overall XGBoost OOF Metrics (FE + TE + 5-fold) ====")
    print(f"OOF Accuracy: {oof_acc:.4f}")
    print(f"OOF F1-score: {oof_f1:.4f}")
    print("\nClassification report (OOF):")
    print(classification_report(y_values, oof_preds, target_names=target_le.classes_))
    print("Confusion matrix (OOF):")
    print(confusion_matrix(y_values, oof_preds))

    # Average test proba
    mean_test_proba = np.mean(np.vstack(test_proba_folds), axis=0)

    # Default threshold=0.5 (you can later do threshold tuning if this is promising)
    test_pred_labels = (mean_test_proba >= 0.5).astype(int)
    test_labels = target_le.inverse_transform(test_pred_labels)

    submission = sample.copy()
    submission[submission.columns[-1]] = test_labels
    submission.to_csv(args.output, index=False)
    print("\nSaved XGBoost FE+TE KFold submission to:", args.output)

    # Optionally save probs for threshold tuning
    np.save("xgb_oof_proba.npy", oof_proba)
    np.save("xgb_test_proba.npy", mean_test_proba)
    print("Saved xgb_oof_proba.npy and xgb_test_proba.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="submission_xgb_fe_te_kfold.csv",
        help="Output submission filename",
    )
    args = parser.parse_args()
    main(args)
