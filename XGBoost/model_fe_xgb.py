# filename: model_all_fe_xgb.py

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")

# ----- Optional boosted libraries -----
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed, skipping XGBClassifier.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("lightgbm not installed, skipping LGBMClassifier.")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("catboost not installed, skipping CatBoostClassifier.\n")

FINAL_SCORES = {}          # model_name -> F1
TRAINED_MODELS = {}        # model_name -> estimator
PREPROCESSOR = None        # fitted ColumnTransformer
FS_TOP_IDX = None          # indices of selected top features


# -------------------- FEATURE ENGINEERING --------------------

def clean_and_engineer(train_df, test_df):
    """
    Do cleaning + feature engineering on raw train & test.
    Return: train_fe, test_fe with same feature columns (train has target too).
    """

    df_all = pd.concat(
        [train_df.assign(_is_train=1), test_df.assign(_is_train=0)],
        ignore_index=True
    )

    # ---------- Basic cleaning ----------
    num_cols_to_impute = ["monthly_revenue_generated", "num_dependents", "years_since_founding"]
    for col in num_cols_to_impute:
        median_val = df_all[col].median()
        df_all[col] = df_all[col].fillna(median_val)

    cat_cols_to_impute = ["work_life_balance_rating", "venture_satisfaction", "team_size_category"]
    for col in cat_cols_to_impute:
        mode_val = df_all[col].mode().iloc[0]
        df_all[col] = df_all[col].fillna(mode_val)

    # ---------- Feature engineering: bins ----------
    age_bins = pd.cut(
        df_all["founder_age"],
        bins=[0, 30, 40, 50, 60, 100],
        include_lowest=True
    )
    df_all["founder_age_bin"] = age_bins.cat.codes

    yws_bins = pd.cut(
        df_all["years_with_startup"],
        bins=[-1, 3, 7, 15, 100],
        include_lowest=True
    )
    df_all["years_with_startup_bin"] = yws_bins.cat.codes

    ysf_bins = pd.cut(
        df_all["years_since_founding"],
        bins=[-1, 2, 5, 10, 100],
        include_lowest=True
    )
    df_all["years_since_founding_bin"] = ysf_bins.cat.codes

    # ---------- Numeric transforms ----------
    df_all["monthly_revenue_log"] = np.log1p(df_all["monthly_revenue_generated"])
    df_all["revenue_per_year"] = df_all["monthly_revenue_generated"] / (df_all["years_since_founding"] + 1.0)
    df_all["dependents_ratio"] = df_all["num_dependents"] / (df_all["founder_age"] + 1.0)

    # tenure ratio: how long founder is with startup vs company age
    df_all["tenure_ratio"] = df_all["years_with_startup"] / (df_all["years_since_founding"] + 1.0)

    # "seniority" approx: age at founding
    df_all["age_at_founding"] = df_all["founder_age"] - df_all["years_since_founding"]

    # age per dependent
    df_all["age_per_dependent"] = df_all["founder_age"] / (df_all["num_dependents"] + 1.0)

    # ---------- Ratings to numeric ----------
    rating_map = {
        "Poor": 0,
        "Fair": 1,
        "Average": 1,
        "Moderate": 1,
        "Good": 2,
        "Very Good": 3,
        "Excellent": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1,
        "Satisfied": 3,
        "Neutral": 2,
        "Dissatisfied": 1
    }

    def map_rating(series):
        return series.map(rating_map).fillna(1).astype(float)

    df_all["wlb_score"] = map_rating(df_all["work_life_balance_rating"])
    df_all["satisfaction_score"] = map_rating(df_all["venture_satisfaction"])
    df_all["performance_score"] = map_rating(df_all["startup_performance_rating"])
    df_all["reputation_score"] = map_rating(df_all["startup_reputation"])
    df_all["visibility_score"] = map_rating(df_all["founder_visibility"])

    # ---------- Binary flags ----------
    for col in ["working_overtime", "remote_operations", "innovation_support"]:
        df_all[col + "_flag"] = (df_all[col].astype(str).str.lower() == "yes").astype(int)

    # remote + overtime mismatch
    df_all["remote_and_overtime"] = df_all["remote_operations_flag"] * df_all["working_overtime_flag"]

    # ---------- Interaction features ----------
    df_all["wlb_overtime_interaction"] = df_all["wlb_score"] * df_all["working_overtime_flag"]
    df_all["satisfaction_performance_interaction"] = df_all["satisfaction_score"] * df_all["performance_score"]
    df_all["reputation_visibility_interaction"] = df_all["reputation_score"] * df_all["visibility_score"]

    # ---------- Frequency encoding for some categories ----------
    freq_cols = [
        "founder_role",
        "education_background",
        "personal_status",
        "startup_stage",
        "team_size_category",
        "founder_gender",
    ]

    n = len(df_all)
    for col in freq_cols:
        vc = df_all[col].value_counts(dropna=False)
        freq_map = (vc / n).to_dict()
        df_all[col + "_freq"] = df_all[col].map(freq_map).fillna(0.0)

    # Split back
    train_fe = df_all[df_all["_is_train"] == 1].drop(columns=["_is_train"])
    test_fe = df_all[df_all["_is_train"] == 0].drop(columns=["_is_train"])

    return train_fe, test_fe


# -------------------- PREPROCESS + FEATURE SELECTION --------------------

def preprocess_and_select_features(train_fe, test_fe, target_col="retention_status", top_k=120):
    """
    - Split X / y
    - Build ColumnTransformer (scale numeric, one-hot categorical)
    - Transform train & test
    - Train RandomForest for feature importance
    - Keep top_k most important features
    Return X_fs, X_test_fs, y_enc, label_encoder
    """

    global PREPROCESSOR, FS_TOP_IDX

    y = train_fe[target_col]
    X = train_fe.drop(columns=[target_col, "founder_id"])
    X_test = test_fe.drop(columns=["founder_id"])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    PREPROCESSOR = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_proc = PREPROCESSOR.fit_transform(X)
    X_test_proc = PREPROCESSOR.transform(X_test)

    # sparse -> dense if needed
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    # Feature selection via RF importance
    fs_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    fs_model.fit(X_proc, y_enc)
    importances = fs_model.feature_importances_

    sorted_idx = np.argsort(importances)[::-1]
    top_k = min(top_k, X_proc.shape[1])
    FS_TOP_IDX = sorted_idx[:top_k]

    X_fs = X_proc[:, FS_TOP_IDX]
    X_test_fs = X_test_proc[:, FS_TOP_IDX]

    return X_fs, X_test_fs, y_enc, le


# -------------------- MODEL EVAL --------------------

def evaluate_model(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    FINAL_SCORES[model_name] = f1
    print(f"{model_name} Accuracy: {acc:.4f}")
    print(f"{model_name} F1 Score: {f1:.4f}")
    print(classification_report(y_true, y_pred))
    print()


# -------------------- PART 1: NORMAL MODELS ON FULL DATA --------------------

def train_normal_models(X, y):
    """
    Normal classification models on full dataset (80/20 split),
    including XGBoost + LightGBM + CatBoost.
    """
    print("\n==== Part 1: Normal Classification Models (80/20, engineered+selected features) ====\n")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "HistGB": HistGradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }

    # Add boosted big guns
    # ---- XGBoost ----
    if HAS_XGB:
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        spw = float(neg) / max(pos, 1)
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1
        )

    # ---- LightGBM ----
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=42
        )

    # ---- CatBoost ----
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            depth=6,
            learning_rate=0.05,
            n_estimators=600,
            verbose=False,
            auto_class_weights="Balanced",
            random_state=42
        )

    for name, model in models.items():
        print(f"--- {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        TRAINED_MODELS[name] = model
        evaluate_model(name, y_val, y_pred)


# -------------------- SUMMARY + SUBMISSION --------------------

def print_final_summary():
    print("\n==================== FINAL F1 SCORE SUMMARY ====================\n")

    best_model = None
    best_f1 = -1.0

    for model_name, f1 in FINAL_SCORES.items():
        print(f"{model_name:25} F1 Score = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_model = model_name

    print("\nBest model based on F1:")
    print(f"  {best_model} with F1 = {best_f1:.4f}")
    print("\n===============================================================\n")

    return best_model, best_f1


def generate_submission(best_model_name, label_encoder, X_test_fs, test_fe):
    print(f"\n=== Generating submission using best model: {best_model_name} ===\n")

    BASE = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(BASE, "submission_all_fe_xgb.csv")

    model = TRAINED_MODELS[best_model_name]

    y_pred_enc = model.predict(X_test_fs)
    y_pred_labels = label_encoder.inverse_transform(y_pred_enc)

    submission = pd.DataFrame({
        "founder_id": test_fe["founder_id"],
        "retention_status": y_pred_labels
    })

    submission.to_csv(out_path, index=False)
    print("submission_all_fe_xgb.csv generated successfully!")
    print(submission.head())


# -------------------- MAIN --------------------

def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(BASE, "train.csv")
    test_path = os.path.join(BASE, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1) Cleaning + feature engineering
    train_fe, test_fe = clean_and_engineer(train_df, test_df)

    # 2) Preprocess + feature selection (top-k features)
    X_fs, X_test_fs, y_enc, label_encoder = preprocess_and_select_features(
        train_fe, test_fe, target_col="retention_status", top_k=120
    )

    # 3) Normal models on full data (now includes XGB, LGBM, CatBoost)
    train_normal_models(X_fs, y_enc)

    # 4) SVM + NN on 20% subset (MANDATORY RULE)
    train_svm_and_nn_20pct(X_fs, y_enc)

    # 5) Summary + submission
    best_model, best_f1 = print_final_summary()
    print(f"Using best model '{best_model}' (F1={best_f1:.4f}) to generate submission.\n")

    generate_submission(best_model, label_encoder, X_test_fs, test_fe)


if __name__ == "__main__":
    main()
