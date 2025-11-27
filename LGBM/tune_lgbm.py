#!/usr/bin/env python3
"""
tune_lgbm.py
Quick manual tuning of LightGBM for founder retention.
Runs several good parameter configs and reports F1/Accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

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
# Main tuning logic
# -------------------------
def main():
    # Load data
    train = pd.read_csv("train.csv")

    y = train["retention_status"]
    X = train.drop(["retention_status"], axis=1)

    # Preprocess
    X_proc, encs = preprocess_fit(X)

    # Encode target
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y)

    # Train/val split (same style as before)
    X_train, X_val, y_train, y_val = train_test_split(
        X_proc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Parameter sets to try (small but strong)
    param_list = [
        {
            "name": "baseline_like",
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "class_weight": None,
        },
        {
            "name": "deeper_leaves",
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "class_weight": None,
        },
        {
            "name": "regularized_depth8",
            "num_leaves": 31,
            "max_depth": 8,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "class_weight": None,
        },
        {
            "name": "more_leaves_depth10",
            "num_leaves": 63,
            "max_depth": 10,
            "min_child_samples": 40,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "class_weight": None,
        },
        {
            "name": "small_leaves_balanced",
            "num_leaves": 31,
            "max_depth": 8,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "class_weight": "balanced",
        },
        {
            "name": "big_leaves_balanced",
            "num_leaves": 63,
            "max_depth": 10,
            "min_child_samples": 40,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "class_weight": "balanced",
        },
    ]

    results = []

    for params in param_list:
        print("\n==============================")
        print("Training config:", params["name"])
        print("==============================")

        clf = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            feature_fraction=params["feature_fraction"],
            bagging_fraction=params["bagging_fraction"],
            class_weight=params["class_weight"],
            random_state=42,
            n_jobs=-1,
        )

        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(period=50),
            ],
        )

        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        best_iter = getattr(clf, "best_iteration_", None)

        print(f"Config: {params['name']}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  Best iteration: {best_iter}")

        result = {
            "name": params["name"],
            "acc": acc,
            "f1": f1,
            "best_iteration": best_iter,
            "params": params,
        }
        results.append(result)

    # Sort and print summary
    print("\n====== SUMMARY (sorted by F1) ======")
    results_sorted = sorted(results, key=lambda x: x["f1"], reverse=True)
    for r in results_sorted:
        print(
            f"{r['name']}: F1={r['f1']:.4f}, Acc={r['acc']:.4f}, "
            f"best_iter={r['best_iteration']}, params={r['params']}"
        )


if __name__ == "__main__":
    main()
