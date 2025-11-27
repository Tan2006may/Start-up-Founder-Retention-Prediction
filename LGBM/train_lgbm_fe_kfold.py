#!/usr/bin/env python3
"""
train_lgbm_fe_kfold.py
LightGBM + same feature engineering + 5-fold CV.
Saves OOF and test probabilities for ensembling.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import lightgbm as lgb

# ---- copy the SAME add_feature_engineering as CatBoost script ----
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

    if "founder_id" in df.columns:
        df = df.drop(columns=["founder_id"])

    return df

def main(args):
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

    target_col = "retention_status"
    y = train[target_col]
    X = train.drop(columns=[target_col])

    # Feature engineering
    X_fe = add_feature_engineering(X)
    test_fe = add_feature_engineering(test)

    # Simple encoding for all categoricals (LabelEncoder folds-wise)
    cat_cols = X_fe.select_dtypes(include=["object", "category"]).columns.tolist()

    # Convert categories to string (for consistency)
    for c in cat_cols:
        X_fe[c] = X_fe[c].astype(str)
        test_fe[c] = test_fe[c].astype(str)

    # Encode target
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y)

    X_values = X_fe.reset_index(drop=True)
    test_values = test_fe.reset_index(drop=True)
    y_values = y_enc

    oof_preds = np.zeros_like(y_values)
    oof_proba = np.zeros(len(y_values), dtype=float)
    test_proba_folds = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_values, y_values), 1):
        print(f"\n===== LGBM Fold {fold} =====")

        X_tr = X_values.iloc[tr_idx].copy()
        y_tr = y_values[tr_idx]
        X_val = X_values.iloc[val_idx].copy()
        y_val = y_values[val_idx]

        # Fit LabelEncoder separately for each categorical col on train fold
        encoders = {}
        for c in cat_cols:
            le = LabelEncoder()
            X_tr[c] = le.fit_transform(X_tr[c])
            # transform val & test with unseen handling
            def map_with_unseen(series):
                mapping = {v: i for i, v in enumerate(le.classes_)}
                return series.map(mapping).fillna(len(le.classes_)).astype(int)
            X_val[c] = map_with_unseen(X_val[c])
            test_values[c] = map_with_unseen(test_values[c])
            encoders[c] = le

        model = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            class_weight=None,
            random_state=42 + fold,
            n_jobs=-1
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )

        best_iter = model.best_iteration_ or 800
        print("Best iteration:", best_iter)

        val_proba = model.predict_proba(X_val, num_iteration=best_iter)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)

        oof_proba[val_idx] = val_proba
        oof_preds[val_idx] = val_pred

        test_proba = model.predict_proba(test_values, num_iteration=best_iter)[:, 1]
        test_proba_folds.append(test_proba)

        fold_acc = accuracy_score(y_val, val_pred)
        fold_f1 = f1_score(y_val, val_pred)
        print(f"Fold {fold} Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}")

    # OOF metrics
    oof_acc = accuracy_score(y_values, oof_preds)
    oof_f1 = f1_score(y_values, oof_preds)

    print("\n==== Overall LGBM OOF Metrics (5-fold + FE) ====")
    print(f"OOF Accuracy: {oof_acc:.4f}")
    print(f"OOF F1-score: {oof_f1:.4f}")
    print("\nClassification report (OOF):")
    print(classification_report(y_values, oof_preds, target_names=target_le.classes_))
    print("Confusion matrix (OOF):")
    print(confusion_matrix(y_values, oof_preds))

    # Save probabilities
    mean_test_proba = np.mean(np.vstack(test_proba_folds), axis=0)
    np.save("lgb_oof_proba.npy", oof_proba)
    np.save("lgb_test_proba.npy", mean_test_proba)
    print("Saved LGBM OOF and test probabilities for ensembling.")

    # Also write a standalone LGBM submission (0.5 threshold) just to see
    test_labels = target_le.inverse_transform((mean_test_proba >= 0.5).astype(int))
    submission = sample.copy()
    submission[submission.columns[-1]] = test_labels
    submission.to_csv(args.output, index=False)
    print("Saved LGBM KFold submission to:", args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="submission_lgbm_fe_kfold.csv",
        help="Output submission filename"
    )
    args = parser.parse_args()
    main(args)
