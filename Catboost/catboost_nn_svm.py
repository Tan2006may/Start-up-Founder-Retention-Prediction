#!/usr/bin/env python3
"""
train_catboost_svm_nn_blend.py

Pipeline:
- Feature engineering (same as your best CatBoost script)
- Split train -> MAIN (80%) and HOLDOUT (20%)
- K-Fold Target Encoding computed ONLY on MAIN, applied to MAIN, HOLDOUT, TEST
- Train CatBoost with K-Fold on MAIN -> produce OOF & test probs
- Train SVM and MLP on HOLDOUT -> produce test probs
- Blend: final_proba = w_cat*cat + w_svm*svm + w_nn*nn  (default 0.8/0.1/0.1)
- Produce final submission CSV

Data files (use these paths in your environment):
- /mnt/data/train.csv
- /mnt/data/test.csv
- /mnt/data/sample_submission.csv

Usage:
    python train_catboost_svm_nn_blend.py --output submission_blend_final.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import joblib

# -------------------------
# Feature Engineering (your best version)
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
            # fill later to avoid using medians before split
    # Fill medians
    for c in num_cols:
        if c in df.columns:
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

    # Ratings mappings
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

    # Interaction features (optional, kept minimal to avoid hurting)
    if {"wlb_score", "satisfaction_score", "performance_score"}.issubset(df.columns):
        df["stress_factor"] = 1 / (df["wlb_score"] + 1)
        df["satisfaction_x_visibility"] = df["satisfaction_score"] * df.get("visibility_score", 0)
        df["performance_x_reputation"] = df["performance_score"] * df.get("reputation_score", 0)

    # Drop ID
    if "founder_id" in df.columns:
        df = df.drop(columns=["founder_id"])

    return df

# -------------------------
# K-Fold Target Encoding on MAIN only
# -------------------------
def kfold_target_encode_main_only(main_df, hold_df, test_df, y_main, cols, n_splits=5, smoothing=10):
    """
    Compute leakage-free K-Fold target encoding using MAIN (train_main) only.
    Returns: main_te, holdout_te, test_te
    main_te: with oof encoded values (length = len(main_df))
    holdout_te: test encodings computed with full-main stats (no leakage)
    test_te: test encodings averaged across folds as before
    """
    main_df = main_df.copy()
    hold_df = hold_df.copy()
    test_df = test_df.copy()
    y = np.array(y_main)
    global_mean = y.mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Prepare storage
    for col in cols:
        if col not in main_df.columns:
            continue
        train_encoded = np.zeros(len(main_df))
        test_encoded_folds = []

        # For fold-wise encoding on main => fill train_encoded
        for tr_idx, val_idx in skf.split(main_df, y):
            X_tr_col = main_df.iloc[tr_idx][col]
            y_tr = y[tr_idx]
            temp = pd.DataFrame({col: X_tr_col, "target": y_tr})
            stats = temp.groupby(col)["target"].agg(["mean", "count"])
            stats["smoothed"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            enc_map = stats["smoothed"]
            val_col = main_df.iloc[val_idx][col]
            train_encoded[val_idx] = val_col.map(enc_map).fillna(global_mean).values
            # test encoding for this fold (applied to test_df)
            test_enc = test_df[col].map(enc_map).fillna(global_mean).values
            test_encoded_folds.append(test_enc)

        test_encoded = np.mean(test_encoded_folds, axis=0)
        main_df[col + "_te"] = train_encoded

        # Now build mapping using FULL main (no fold) to map holdout
        full_stats = pd.DataFrame({col: main_df[col], "target": y}).groupby(col)["target"].agg(["mean", "count"])
        full_stats["smoothed"] = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (full_stats["count"] + smoothing)
        full_map = full_stats["smoothed"]
        hold_df[col + "_te"] = hold_df[col].map(full_map).fillna(global_mean).values

        # test already done above via folds
        test_df[col + "_te"] = test_encoded

    return main_df, hold_df, test_df

# -------------------------
# Helper to prepare sklearn-friendly numeric matrices
# -------------------------
def prepare_sklearn_matrix(train_dfs_for_encoding, X_df, categorical_cols):
    """
    Fit LabelEncoders for categorical_cols on the concatenation of provided frames,
    then transform X_df into numeric array (including TE columns and numeric features).
    train_dfs_for_encoding: list of DataFrames to fit encoders on (e.g., [main, holdout, test])
    X_df: dataframe to transform
    categorical_cols: list of categorical column names to encode (non-TE)
    """
    from sklearn.preprocessing import LabelEncoder

    df_cat_fit = pd.concat(train_dfs_for_encoding, axis=0)[categorical_cols].astype(str).fillna("Unknown")
    encoders = {}
    for c in categorical_cols:
        le = LabelEncoder()
        le.fit(df_cat_fit[c])
        encoders[c] = le

    X_copy = X_df.copy()
    for c in categorical_cols:
        X_copy[c] = encoders[c].transform(X_copy[c].astype(str).fillna("Unknown"))

    # Now return numeric matrix: include all numeric dtypes + any *_te columns
    te_cols = [c for c in X_copy.columns if c.endswith("_te")]
    numeric_cols = X_copy.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # ensure TE columns included even if dtype object
    for c in te_cols:
        if c not in numeric_cols:
            numeric_cols.append(c)
    return X_copy[numeric_cols], encoders

# -------------------------
# Main
# -------------------------
def main(args):
    # load
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")


    target_col = "retention_status"
    y = train[target_col]
    X = train.drop(columns=[target_col])

    # Encode target
    target_le = LabelEncoder()
    y_enc_full = target_le.fit_transform(y)

    # Feature engineering on full frame first (so bins & derived cols exist consistently)
    X_fe_all = add_feature_engineering(X)
    test_fe_all = add_feature_engineering(test)

    # Split MAIN (80%) and HOLDOUT (20%) — stratified
    # We'll take one fold from StratifiedKFold as holdout (~20%)
    skf_tmp = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf_tmp.split(X_fe_all, y_enc_full))
    _, hold_idx = folds[0]  # use first fold as holdout
    mask = np.zeros(len(X_fe_all), dtype=bool)
    mask[hold_idx] = True

    X_hold = X_fe_all[mask].reset_index(drop=True)
    y_hold = y_enc_full[mask]
    X_main = X_fe_all[~mask].reset_index(drop=True)
    y_main = y_enc_full[~mask]

    print(f"MAIN size: {len(X_main)}, HOLDOUT size: {len(X_hold)}")

    # Target encoding columns
    te_cols = [
        "founder_role",
        "education_background",
        "personal_status",
        "startup_stage",
        "team_size_category",
    ]

    # Compute K-Fold TE using MAIN only (produces main_te, hold_te, test_te)
    X_main_te, X_hold_te, test_te = kfold_target_encode_main_only(X_main.copy(), X_hold.copy(), test_fe_all.copy(), y_main, te_cols, n_splits=5, smoothing=10)

    # Prepare categorical columns list for CatBoost (string dtype, no NaN)
    cat_cols = X_main_te.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        X_main_te[c] = X_main_te[c].astype("string").fillna("Unknown")
        X_hold_te[c] = X_hold_te[c].astype("string").fillna("Unknown")
        test_te[c] = test_te[c].astype("string").fillna("Unknown")

    # -------------------------
    # Train CatBoost on MAIN with K-Fold (5-fold)
    # -------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_main_te), dtype=int)
    oof_proba = np.zeros(len(X_main_te), dtype=float)
    test_proba_folds = []
    cat_feature_indices = [X_main_te.columns.get_loc(c) for c in cat_cols]
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_main_te, y_main), 1):
        print(f"\n=== CatBoost MAIN Fold {fold} ===")
        X_tr = X_main_te.iloc[train_idx]
        y_tr = y_main[train_idx]
        X_val = X_main_te.iloc[val_idx]
        y_val = y_main[val_idx]

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

        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)

        oof_proba[val_idx] = val_proba
        oof_preds[val_idx] = val_pred

        test_pool = Pool(test_te, cat_features=cat_feature_indices)
        test_proba = model.predict_proba(test_pool)[:, 1]
        test_proba_folds.append(test_proba)

        fold_models.append(model)

        print(f"Fold {fold} Acc: {accuracy_score(y_val, val_pred):.4f}, F1: {f1_score(y_val, val_pred):.4f}")

    # OOF metrics for MAIN
    oof_acc_main = accuracy_score(y_main, oof_preds)
    oof_f1_main = f1_score(y_main, oof_preds)
    print("\n=== CatBoost MAIN OOF Metrics ===")
    print(f"OOF Acc (MAIN): {oof_acc_main:.4f}, OOF F1 (MAIN): {oof_f1_main:.4f}")
    print(classification_report(y_main, oof_preds))
    print(confusion_matrix(y_main, oof_preds))

    mean_test_proba_cat = np.mean(np.vstack(test_proba_folds), axis=0)
    np.save("cat_main_oof_proba.npy", oof_proba)
    np.save("cat_main_test_proba.npy", mean_test_proba_cat)
    print("Saved CatBoost MAIN OOF/test probs.")

    # Also optionally train a final CatBoost on full MAIN for model pickling
    final_cat = CatBoostClassifier(
        iterations=(model.get_best_iteration() or 1000),
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=999,
        verbose=100,
    )
    final_cat.fit(Pool(X_main_te, y_main, cat_features=cat_feature_indices))
    joblib.dump(final_cat, "cat_final_on_main.pkl")
    print("Saved final CatBoost (trained on full MAIN): cat_final_on_main.pkl")

    # -------------------------
    # Prepare HOLDOUT data for sklearn models (SVM, MLP)
    # -------------------------
    # For sklearn, we need numeric matrix. We'll label-encode remaining categoricals using combined MAIN+HOLDOUT+TEST for consistency.
    combined_for_encoding = pd.concat([X_main_te, X_hold_te, test_te], axis=0)
    # Identify categorical columns to encode (exclude TE columns)
    cat_cols_for_sklearn = combined_for_encoding.select_dtypes(include=["object", "category"]).columns.tolist()
    # Ensure TE columns are kept numeric and present
    # Prepare encoders and transform
    def prepare_sklearn_matrix_local(train_dfs_for_encoding, X_df, categorical_cols):
        from sklearn.preprocessing import LabelEncoder
        df_cat_fit = pd.concat(train_dfs_for_encoding, axis=0)[categorical_cols].astype(str).fillna("Unknown")
        encoders = {}
        for c in categorical_cols:
            le = LabelEncoder()
            le.fit(df_cat_fit[c])
            encoders[c] = le

        X_copy = X_df.copy()
        for c in categorical_cols:
            X_copy[c] = encoders[c].transform(X_copy[c].astype(str).fillna("Unknown"))

        te_cols_local = [c for c in X_copy.columns if c.endswith("_te")]
        numeric_cols = X_copy.select_dtypes(include=["int64", "float64"]).columns.tolist()
        for c in te_cols_local:
            if c not in numeric_cols:
                numeric_cols.append(c)
        return X_copy[numeric_cols], encoders

    X_hold_skl, encoders = prepare_sklearn_matrix_local([X_main_te, X_hold_te, test_te], X_hold_te, cat_cols_for_sklearn)
    test_skl, _ = prepare_sklearn_matrix_local([X_main_te, X_hold_te, test_te], test_te, cat_cols_for_sklearn)

    # Impute missing values then standardize for SVM and NN
    imputer = SimpleImputer(strategy="median")
    X_hold_skl = imputer.fit_transform(X_hold_skl)   # fit on HOLDOUT only (as required)
    test_skl = imputer.transform(test_skl)

    scaler = StandardScaler()
    X_hold_skl = scaler.fit_transform(X_hold_skl)
    test_skl = scaler.transform(test_skl)

    # -------------------------
    # Train SVM on HOLDOUT
    # -------------------------
    print("\n=== Training SVM on HOLDOUT ===")
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_hold_skl, y_hold)
    svm_test_proba = svm.predict_proba(test_skl)[:, 1]
    joblib.dump(svm, "svm_holdout.pkl")
    print("Saved svm_holdout.pkl")

    # Evaluate SVM on HOLDOUT (self-check)
    svm_hold_pred = svm.predict(X_hold_skl)
    print("SVM holdout Acc:", accuracy_score(y_hold, svm_hold_pred), "F1:", f1_score(y_hold, svm_hold_pred))

    # -------------------------
    # Train Neural Network (MLP) on HOLDOUT
    # -------------------------
    print("\n=== Training MLP on HOLDOUT ===")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    mlp.fit(X_hold_skl, y_hold)
    mlp_test_proba = mlp.predict_proba(test_skl)[:, 1]
    joblib.dump(mlp, "mlp_holdout.pkl")
    print("Saved mlp_holdout.pkl")

    mlp_hold_pred = mlp.predict(X_hold_skl)
    print("MLP holdout Acc:", accuracy_score(y_hold, mlp_hold_pred), "F1:", f1_score(y_hold, mlp_hold_pred))

    # -------------------------
    # Blend predictions
    # -------------------------
    w_cat = args.w_cat
    w_svm = args.w_svm
    w_nn = args.w_nn
    print(f"\nUsing blend weights cat:{w_cat}, svm:{w_svm}, nn:{w_nn} (must sum to 1.0)")

    blended_test_proba = w_cat * mean_test_proba_cat + w_svm * svm_test_proba + w_nn * mlp_test_proba
    blended_test_labels = (blended_test_proba >= args.threshold).astype(int)
    blended_test_decoded = target_le.inverse_transform(blended_test_labels)

    # Save blended probs
    np.save("blended_test_proba.npy", blended_test_proba)
    np.save("blended_oof_cat_on_main_proba.npy", oof_proba)

    # Build final submission
    submission = sample.copy()
    submission[submission.columns[-1]] = blended_test_decoded
    submission.to_csv(args.output, index=False)
    print("\nSaved final blended submission to:", args.output)

    # Print OOF metrics of CatBoost (MAIN) and holdout model checks
    print("\nSUMMARY:")
    print(f"CatBoost (MAIN) OOF F1: {oof_f1_main:.4f}, Acc: {oof_acc_main:.4f}")
    print("SVM holdout and MLP holdout printed above for quick sanity checks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="submission_blend_final.csv", help="Output filename")
    parser.add_argument("--w_cat", type=float, default=0.8, help="weight for CatBoost in blend")
    parser.add_argument("--w_svm", type=float, default=0.1, help="weight for SVM in blend")
    parser.add_argument("--w_nn", type=float, default=0.1, help="weight for NN in blend")
    parser.add_argument("--threshold", type=float, default=0.5, help="decision threshold")
    args = parser.parse_args()
    main(args)
