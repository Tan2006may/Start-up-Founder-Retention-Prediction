#!/usr/bin/env python3
"""
train_catboost_fe_kfold.py
Feature Engineering + CatBoost with Stratified K-Fold CV.
Generates:
- Out-of-fold validation metrics
- Averaged test predictions submission
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
import joblib

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

    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Ratio features
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

    # Ordinal-like scores
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
        "medium ": 2,
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

    # Binning
    if "founder_age" in df.columns:
        df["founder_age_bin"] = pd.cut(
            df["founder_age"],
            bins=[0, 25, 35, 45, 60, 100],
            labels=["very_young", "young", "mid", "senior", "old"],
            include_lowest=True
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
            include_lowest=True
        )

    # Drop ID-like columns
    if "founder_id" in df.columns:
        df = df.drop(columns=["founder_id"])

    return df

# -------------------------
# Main
# -------------------------
def main(args):
    # Load
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

    target_col = "retention_status"
    y = train[target_col]
    X = train.drop(columns=[target_col])

    # Feature engineering
    X_fe = add_feature_engineering(X)
    test_fe = add_feature_engineering(test)

    # Categorical columns for CatBoost
    cat_cols = X_fe.select_dtypes(include=["object", "category"]).columns.tolist()

    # Ensure categorical columns are strings with no NaN
    for c in cat_cols:
        X_fe[c] = X_fe[c].astype("string").fillna("Unknown")
        test_fe[c] = test_fe[c].astype("string").fillna("Unknown")

    # Encode target
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y)

    # Prepare arrays for OOF predictions
    oof_preds = np.zeros_like(y_enc)
    oof_proba = np.zeros(len(y_enc), dtype=float)

    # For test predictions: collect probabilities & average
    test_proba_folds = []

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    X_values = X_fe.reset_index(drop=True)
    test_values = test_fe.reset_index(drop=True)
    y_values = y_enc

    cat_feature_indices = [X_values.columns.get_loc(c) for c in cat_cols]

    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_values, y_values), 1):
        print(f"\n===== Fold {fold} =====")

        X_tr = X_values.iloc[train_idx]
        y_tr = y_values[train_idx]
        X_val = X_values.iloc[val_idx]
        y_val = y_values[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices)

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=42 + fold,
            od_type="Iter",
            od_wait=60,
            verbose=100
        )

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        # Store model
        fold_models.append(model)

        # OOF predictions
        val_pred = model.predict(X_val)
        val_pred = val_pred.astype(int)
        oof_preds[val_idx] = val_pred

        # OOF probabilities (for info, not strictly needed)
        val_proba = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_proba

        # Test probabilities for this fold
        test_pool = Pool(test_values, cat_features=cat_feature_indices)
        test_proba = model.predict_proba(test_pool)[:, 1]
        test_proba_folds.append(test_proba)

        # Fold metrics
        fold_acc = accuracy_score(y_val, val_pred)
        fold_f1 = f1_score(y_val, val_pred)
        print(f"Fold {fold} Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}")

    # Overall OOF metrics
    oof_acc = accuracy_score(y_values, oof_preds)
    oof_f1 = f1_score(y_values, oof_preds)

        # -------------------------
    # Threshold Optimization
    # -------------------------
    print("\n==== Threshold Optimization ====")

    thresholds = np.arange(0.30, 0.71, 0.01)
    best_thresh = 0.5
    best_f1 = oof_f1

    for t in thresholds:
        preds_t = (oof_proba >= t).astype(int)
        f1_t = f1_score(y_values, preds_t)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t

    print(f"Best threshold: {best_thresh:.2f}, F1: {best_f1:.6f}")


    print("\n==== Overall OOF Metrics (5-fold CatBoost + FE) ====")
    print(f"OOF Accuracy: {oof_acc:.4f}")
    print(f"OOF F1-score: {oof_f1:.4f}")
    print("\nClassification report (OOF):")
    print(classification_report(y_values, oof_preds, target_names=target_le.classes_))
    print("Confusion matrix (OOF):")
    print(confusion_matrix(y_values, oof_preds))

    # Average test probabilities across folds
    mean_test_proba = np.mean(np.vstack(test_proba_folds), axis=0)
    test_pred_labels = (mean_test_proba >= best_thresh).astype(int)
    test_labels = target_le.inverse_transform(test_pred_labels)

    # Build submission
    submission = sample.copy()
    submission[submission.columns[-1]] = test_labels
    submission.to_csv(args.output, index=False)
    print("\nSaved K-fold CatBoost + FE submission to:", args.output)

        # Save OOF and test probabilities for ensembling
    np.save("cat_oof_proba.npy", oof_proba)
    np.save("cat_test_proba.npy", mean_test_proba)
    print("Saved CatBoost OOF and test probabilities for ensembling.")


    # Optionally save models
    for i, m in enumerate(fold_models, 1):
        fname = f"catboost_fe_fold{i}.pkl"
        joblib.dump(m, fname)
        print("Saved fold model:", fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="submission_catboost_fe_kfold.csv",
        help="Output submission filename"
    )
    args = parser.parse_args()
    main(args)
