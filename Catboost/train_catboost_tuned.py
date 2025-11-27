#!/usr/bin/env python3
"""
train_catboost_fe_kfold.py
Feature Engineering + K-Fold Target Encoding + CatBoost K-Fold CV.

Outputs:
- Overall OOF metrics (Accuracy, F1, report, confusion matrix)
- Saved OOF probabilities: cat_oof_proba.npy
- Saved test probabilities: cat_test_proba.npy
- Submission CSV: submission_catboost_fe_kfold_te.csv (or --output name)
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
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

    # =============================
    # Base Ratios (already used)
    # =============================
    df["experience_ratio"] = df["years_with_startup"] / (df["founder_age"] + 1)
    df["funding_per_year"] = df["funding_rounds_led"] / (df["years_since_founding"] + 1)
    df["revenue_per_year"] = df["monthly_revenue_generated"] / (df["years_since_founding"] + 1)
    df["dependents_ratio"] = df["num_dependents"] / (df["founder_age"] + 1)
    df["distance_adjusted"] = df["distance_from_investor_hub"] / (df["years_since_founding"] + 1)

    # =============================
    # Rating Mappings
    # =============================
    def map_rating(series, mapping):
        return series.astype(str).str.strip().str.lower().map(mapping).astype(float)

    rating_map = {"poor":1, "average":2, "medium":2, "good":3, "excellent":4}
    sat_map    = {"low":1, "medium":2, "neutral":2, "high":3, "satisfied":3}
    rep_map    = {"poor":1, "fair":2, "moderate":2, "good":3, "excellent":4, "high":4, "low":1}
    vis_map    = {"low":1, "medium":2, "high":3}

    df["wlb_score"] = map_rating(df["work_life_balance_rating"], rating_map)
    df["satisfaction_score"] = map_rating(df["venture_satisfaction"], sat_map)
    df["performance_score"] = map_rating(df["startup_performance_rating"], rating_map)
    df["reputation_score"] = map_rating(df["startup_reputation"], rep_map)
    df["visibility_score"] = map_rating(df["founder_visibility"], vis_map)

    # =============================
    # NEW INTERACTION FEATURES
    # =============================
    df["stress_factor"] = 1 / (df["wlb_score"] + 1)
    df["satisfaction_x_visibility"] = df["satisfaction_score"] * df["visibility_score"]
    df["performance_x_reputation"] = df["performance_score"] * df["reputation_score"]
    df["revenue_pressure"] = df["revenue_per_year"] / (df["team_size_category"].factorize()[0] + 1)
    df["investment_distance_risk"] = df["distance_adjusted"] * df["performance_score"]

    # =============================
    # Domain Bucketing
    # =============================
    df["high_risk_flag"] = (
        (df["wlb_score"] <= 2) &
        (df["satisfaction_score"] <= 2) &
        (df["performance_score"] <= 2)
    ).astype(int)

    df["high_stable_flag"] = (
        (df["wlb_score"] >= 3) &
        (df["satisfaction_score"] >= 2) &
        (df["performance_score"] >= 3)
    ).astype(int)

    # Drop ID
    if "founder_id" in df.columns:  
        df = df.drop(columns=["founder_id"])

    return df


# -------------------------
# K-Fold Target Encoding
# -------------------------
def kfold_target_encode(train_df, test_df, y, cols, n_splits=5, smoothing=10):
    """
    Leakage-safe K-Fold target encoding.
    Args:
        train_df: feature dataframe (no target column)
        test_df: test feature dataframe
        y: numpy array or Series of encoded target (0/1)
        cols: list of column names to target-encode
    Returns:
        train_df_te, test_df_te with new *_te columns added
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
        test_encoded_folds = []

        for tr_idx, val_idx in skf.split(train_df, y):
            X_tr_col = train_df.iloc[tr_idx][col]
            y_tr = y[tr_idx]

            temp = pd.DataFrame({col: X_tr_col, "target": y_tr})
            stats = temp.groupby(col)["target"].agg(["mean", "count"])
            stats["smoothed"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
                stats["count"] + smoothing
            )
            enc_map = stats["smoothed"]

            # validation fold
            val_col = train_df.iloc[val_idx][col]
            train_encoded[val_idx] = val_col.map(enc_map).fillna(global_mean).values

            # test encoded for this fold
            test_enc = test_df[col].map(enc_map).fillna(global_mean).values
            test_encoded_folds.append(test_enc)

        test_encoded = np.mean(test_encoded_folds, axis=0)

        train_df[col + "_te"] = train_encoded
        test_df[col + "_te"] = test_encoded

    return train_df, test_df

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

    # Encode target first (0/1)
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y)

    # Feature engineering
    X_fe = add_feature_engineering(X)
    test_fe = add_feature_engineering(test)

    # Target encoding on selected high-signal categoricals
    target_encode_cols = [
        "founder_role",
        "education_background",
        "personal_status",
        "startup_stage",
        "team_size_category",
    ]

    X_fe_te, test_fe_te = kfold_target_encode(
        X_fe, test_fe, y_enc, target_encode_cols, n_splits=5, smoothing=10
    )

    # Categorical columns for CatBoost (string, no NaN)
    cat_cols = X_fe_te.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        X_fe_te[c] = X_fe_te[c].astype("string").fillna("Unknown")
        test_fe_te[c] = test_fe_te[c].astype("string").fillna("Unknown")

    X_values = X_fe_te.reset_index(drop=True)
    test_values = test_fe_te.reset_index(drop=True)
    y_values = y_enc

    # Prepare OOF storage
    oof_preds = np.zeros_like(y_values)
    oof_proba = np.zeros(len(y_values), dtype=float)
    test_proba_folds = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cat_feature_indices = [X_values.columns.get_loc(c) for c in cat_cols]

    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_values, y_values), 1):
        print(f"\n===== CatBoost Fold {fold} =====")

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
            verbose=100,
        )

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        fold_models.append(model)

        val_pred = model.predict(X_val)
        val_pred = val_pred.astype(int)
        val_proba = model.predict_proba(X_val)[:, 1]

        oof_preds[val_idx] = val_pred
        oof_proba[val_idx] = val_proba

        test_pool = Pool(test_values, cat_features=cat_feature_indices)
        test_proba = model.predict_proba(test_pool)[:, 1]
        test_proba_folds.append(test_proba)

        fold_acc = accuracy_score(y_val, val_pred)
        fold_f1 = f1_score(y_val, val_pred)
        print(f"Fold {fold} Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}")

    # OOF metrics
    oof_acc = accuracy_score(y_values, oof_preds)
    oof_f1 = f1_score(y_values, oof_preds)
    print("\n==== Overall OOF Metrics (CatBoost + FE + TE, 5-fold) ====")
    print(f"OOF Accuracy: {oof_acc:.4f}")
    print(f"OOF F1-score: {oof_f1:.4f}")
    print("\nClassification report (OOF):")
    print(classification_report(y_values, oof_preds, target_names=target_le.classes_))
    print("Confusion matrix (OOF):")
    print(confusion_matrix(y_values, oof_preds))

    # Save OOF & test probabilities
    mean_test_proba = np.mean(np.vstack(test_proba_folds), axis=0)
    np.save("cat_oof_proba.npy", oof_proba)
    np.save("cat_test_proba.npy", mean_test_proba)
    print("Saved CatBoost OOF and test probabilities for ensembling.")

    # Default 0.5 threshold for submission (you can later re-use best threshold)
    test_pred_labels = (mean_test_proba >= 0.5).astype(int)
    test_labels = target_le.inverse_transform(test_pred_labels)

    submission = sample.copy()
    submission[submission.columns[-1]] = test_labels
    submission.to_csv(args.output, index=False)
    print("\nSaved K-fold CatBoost + FE + TE submission to:", args.output)

    # Save fold models
    for i, m in enumerate(fold_models, 1):
        fname = f"catboost_fe_te_fold{i}.pkl"
        joblib.dump(m, fname)
        print("Saved fold model:", fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="submission_catboost_fe_kfold_te.csv",
        help="Output submission filename",
    )
    args = parser.parse_args()
    main(args)
