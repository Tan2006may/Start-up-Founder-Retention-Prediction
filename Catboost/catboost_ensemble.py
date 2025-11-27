#!/usr/bin/env python3
"""
train_catboost_ensemble.py
5-model CatBoost ensemble with:
- Feature Engineering
- K-Fold Target Encoding
- 5 different seeds / small param tweaks
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostClassifier, Pool
import joblib

# -------------------------
# Feature Engineering (same style as your best TE model)
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

    # Basic ratios
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

    # Ordinal mappings
    def map_rating(series, mapping):
        return series.astype(str).str.strip().str.lower().map(mapping).astype(float)

    rating_map = {"poor":1, "average":2, "medium":2, "good":3, "excellent":4}
    sat_map    = {"low":1, "medium":2, "neutral":2, "high":3, "satisfied":3}
    rep_map    = {"poor":1, "fair":2, "moderate":2, "good":3, "excellent":4, "high":4, "low":1}
    vis_map    = {"low":1, "medium":2, "high":3}

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

    # Drop ID
    if "founder_id" in df.columns:
        df = df.drop(columns=["founder_id"])

    return df

# -------------------------
# K-Fold Target Encoding
# -------------------------
def kfold_target_encode(train_df, test_df, y, cols, n_splits=5, smoothing=10):
    train_df = train_df.copy()
    test_df = test_df.copy()
    y = np.array(y)
    global_mean = y.mean()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in cols:
        if col not in train_df.columns:
            continue

        train_encoded = np.zeros(len(train_df))
        test_folds = []

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
            test_folds.append(test_df[col].map(enc_map).fillna(global_mean).values)

        test_encoded = np.mean(test_folds, axis=0)
        train_df[col + "_te"] = train_encoded
        test_df[col + "_te"] = test_encoded

    return train_df, test_df

# -------------------------
# Train one model (K-Fold) and return OOF + test proba
# -------------------------
def train_single_model(X_values, test_values, y_values, cat_feature_indices, seed, extra_params=None):
    base_params = dict(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=seed,
        od_type="Iter",
        od_wait=60,
        verbose=100,
    )
    if extra_params:
        base_params.update(extra_params)

    oof_proba = np.zeros(len(X_values))
    test_proba_folds = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_values, y_values), 1):
        print(f"  Fold {fold} (seed={seed})")

        X_tr = X_values.iloc[tr_idx]
        y_tr = y_values[tr_idx]
        X_val = X_values.iloc[val_idx]
        y_val = y_values[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices)

        model = CatBoostClassifier(**base_params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_proba = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_proba

        test_pool = Pool(test_values, cat_features=cat_feature_indices)
        test_proba = model.predict_proba(test_pool)[:, 1]
        test_proba_folds.append(test_proba)

    mean_test_proba = np.mean(test_proba_folds, axis=0)
    return oof_proba, mean_test_proba

# -------------------------
# Main
# -------------------------
def main(args):
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

    y = train["retention_status"]
    X = train.drop(columns=["retention_status"])

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # FE
    X_fe = add_feature_engineering(X)
    test_fe = add_feature_engineering(test)

    # Target encoding
    te_cols = ["founder_role", "education_background", "personal_status", "startup_stage", "team_size_category"]
    X_te, test_te = kfold_target_encode(X_fe, test_fe, y_enc, te_cols)

    # Detect categoricals BEFORE converting to string
    cat_cols = X_te.select_dtypes(include=["object", "category"]).columns.tolist()

    # Convert categoricals to string (CatBoost likes string/int)
    for c in cat_cols:
        X_te[c] = X_te[c].astype("string").fillna("Unknown")
        test_te[c] = test_te[c].astype("string").fillna("Unknown")

    X_values = X_te.reset_index(drop=True)
    test_values = test_te.reset_index(drop=True)
    y_values = y_enc

    # Indices of categorical features (fixed)
    cat_feature_indices = [X_values.columns.get_loc(c) for c in cat_cols]

    # Define ensemble models (seed + tweaks)
    models = [
        (42,   {}),                 # base
        (73,   {}),                 # alt seed
        (111,  {}),                 # alt seed
        (2024, {"depth": 7}),       # slightly deeper tree
        (4040, {"bagging_temperature": 0.4}),  # more sampling variety
    ]

    all_oof = []
    all_test = []

    for i, (seed, extra) in enumerate(models, 1):
        print(f"\n=== Training Ensemble Model {i} (seed={seed}, extra={extra}) ===")
        oof_proba, test_proba = train_single_model(
            X_values, test_values, y_values, cat_feature_indices, seed, extra
        )
        all_oof.append(oof_proba)
        all_test.append(test_proba)

    # Blend OOF and test probs
    oof_blend = np.mean(all_oof, axis=0)
    test_blend = np.mean(all_test, axis=0)

    # Evaluate OOF
    oof_preds = (oof_blend >= 0.5).astype(int)
    oof_f1 = f1_score(y_values, oof_preds)
    oof_acc = accuracy_score(y_values, oof_preds)
    print("\n===== ENSEMBLE OOF METRICS =====")
    print(f"Accuracy: {oof_acc:.4f}")
    print(f"F1-score: {oof_f1:.4f}")

    # Submission
    test_labels = le.inverse_transform((test_blend >= 0.5).astype(int))
    sub = sample.copy()
    sub[sub.columns[-1]] = test_labels
    sub.to_csv(args.output, index=False)
    print("\nSaved ensemble submission to:", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="submission_catboost_ensemble.csv")
    args = parser.parse_args()
    main(args)
