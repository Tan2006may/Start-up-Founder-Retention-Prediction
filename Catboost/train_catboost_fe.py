#!/usr/bin/env python3
"""
train_catboost_fe.py
Feature engineering + CatBoost model for founder retention.
- Does train/validation eval
- Then trains on full data
- Generates submission CSV
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
import joblib

# -------------------------
# Feature Engineering
# -------------------------
def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Basic safe numeric conversions
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

    # Fill some numeric NaNs early for ratios
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # 1) Ratio / efficiency features
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

    # 2) Ordinal encodings as extra numeric features
    def map_rating(series, mapping):
        return series.astype(str).str.strip().str.lower().map(mapping).astype(float)

    # Example maps (robust to missing/unseen -> NaN)
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

    if "work_life_balance_rating" in df.columns:
        df["wlb_score"] = map_rating(df["work_life_balance_rating"], rating_map)

    if "venture_satisfaction" in df.columns:
        df["satisfaction_score"] = map_rating(df["venture_satisfaction"], sat_map)

    if "startup_performance_rating" in df.columns:
        df["performance_score"] = map_rating(df["startup_performance_rating"], rating_map)

    if "startup_reputation" in df.columns:
        df["reputation_score"] = map_rating(df["startup_reputation"], rep_map)

    if "founder_visibility" in df.columns:
        vis_map = {"low": 1, "medium": 2, "high": 3}
        df["visibility_score"] = map_rating(df["founder_visibility"], vis_map)

    # 3) Binning for some continuous features
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

    # Drop pure ID columns
    if "founder_id" in df.columns:
        df = df.drop(columns=["founder_id"])

    return df

# -------------------------
# Main training function
# -------------------------
def main(args):
    # Load data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

    # Separate target
    target_col = "retention_status"
    y = train[target_col]
    X = train.drop(columns=[target_col])

    # Feature engineering
    X_fe = add_feature_engineering(X)
    test_fe = add_feature_engineering(test)

    # Identify categorical columns for CatBoost
        # Identify categorical columns for CatBoost
    cat_cols = X_fe.select_dtypes(include=["object", "category"]).columns.tolist()

    # CatBoost requirement: categorical values must be strings, no NaN
    for c in cat_cols:
        X_fe[c] = X_fe[c].astype("string").fillna("Unknown")
        test_fe[c] = test_fe[c].astype("string").fillna("Unknown")


    # Encode target
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_fe, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Indices of categorical features
    cat_features_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    # Pools for CatBoost
    train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_indices)

    # CatBoost model (initial config – can be tuned)
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=42,
        od_type="Iter",
        od_wait=60,
        verbose=100
    )

    print("Training CatBoost with train/val split...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Validation metrics
    val_pred = model.predict(X_val)
    val_pred = val_pred.astype(int)  # CatBoost returns labels (0/1) as strings sometimes

    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)

    print("\n==== Validation Metrics (CatBoost + FE) ====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, val_pred, target_names=target_le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, val_pred))

    # -------- Train on FULL data for submission --------
    print("\nTraining final CatBoost model on FULL training data...")

    full_pool = Pool(X_fe, y_enc, cat_features=cat_features_indices)

    final_model = CatBoostClassifier(
        iterations=model.get_best_iteration() or 1000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=42,
        verbose=100
    )
    final_model.fit(full_pool)

    # Predict on test
    print("Predicting on test data...")
    test_pool = Pool(test_fe, cat_features=cat_features_indices)
    test_pred = final_model.predict(test_pool)
    test_pred = test_pred.astype(int)

    test_labels = target_le.inverse_transform(test_pred)

    # Build submission
    submission = sample.copy()
    submission[submission.columns[-1]] = test_labels
    submission.to_csv(args.output, index=False)
    print("\nSaved CatBoost + FE submission to:", args.output)

    # Save model
    joblib.dump(final_model, "catboost_fe_model.pkl")
    print("Saved final CatBoost model to: catboost_fe_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="submission_catboost_fe.csv",
        help="Output submission filename"
    )
    args = parser.parse_args()
    main(args)
