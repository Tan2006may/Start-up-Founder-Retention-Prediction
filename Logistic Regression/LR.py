import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
ID_COL = "founder_id"
TARGET_COL = "retention_status"
OUTPUT_CSV_PREFIX = "submission"

# -------------------------
# Utility functions
# -------------------------
def best_threshold(y_true, y_proba, min_thr=0.1, max_thr=0.9, step=0.01):
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(min_thr, max_thr + 1e-8, step):
        preds = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1

def clean_numeric(df, num_cols, medians=None):
    df_num = df[num_cols].copy()
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = df_num.median()
    df_num = df_num.fillna(medians)
    return df_num, medians

def build_freq_encoding(series):
    return series.value_counts(normalize=True)

def apply_freq_encoding(series, freq_map):
    return series.map(freq_map).fillna(0.0)

# -------------------------
# Load + preprocess
# -------------------------
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if TARGET_COL not in train.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in train.csv")
    
    y_raw = train[TARGET_COL]
    X_raw = train.drop(columns=[TARGET_COL], errors="ignore")
    
    # Drop ID from features
    if ID_COL in X_raw.columns:
        X_raw = X_raw.drop(columns=[ID_COL], errors="ignore")
    test_ids = test[ID_COL] if ID_COL in test.columns else pd.Series(range(len(test)))
    X_test_raw = test.drop(columns=[ID_COL], errors="ignore")
    
    # Align columns
    common_cols = [c for c in X_raw.columns if c in X_test_raw.columns]
    X_raw = X_raw[common_cols].copy()
    X_test_raw = X_test_raw[common_cols].copy()
    
    # Map target to 0/1 but remember original labels (e.g. 'Left', 'Stayed')
    y_unique = sorted(y_raw.dropna().unique())
    if len(y_unique) != 2:
        raise ValueError(f"Expected binary target, found: {y_unique}")
    if set(y_unique) == {0, 1}:
        label_to_int = {0: 0, 1: 1}
    else:
        label_to_int = {y_unique[0]: 0, y_unique[1]: 1}
    int_to_label = {v: k for k, v in label_to_int.items()}
    y = y_raw.map(label_to_int).astype(int)
    
    return X_raw, y, X_test_raw, test_ids, label_to_int, int_to_label

def preprocess_for_models(X, X_test):
    # Identify categorical / numeric
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    # ----- Numeric+freq for Logistic Regression -----
    X_ml = pd.DataFrame(index=X.index)
    X_test_ml = pd.DataFrame(index=X_test.index)
    
    # numeric
    X_num, med_ml = clean_numeric(X, num_cols)
    X_test_num, _ = clean_numeric(X_test, num_cols, medians=med_ml)
    
    for col in num_cols:
        X_ml[col] = X_num[col]
        X_test_ml[col] = X_test_num[col]
    
    # freq-encoded categoricals
    for col in cat_cols:
        freq_map = build_freq_encoding(X[col].astype(str).fillna("Missing"))
        X_ml[col + "_freq"] = apply_freq_encoding(X[col].astype(str).fillna("Missing"), freq_map)
        X_test_ml[col + "_freq"] = apply_freq_encoding(
            X_test[col].astype(str).fillna("Missing"), freq_map)
    
    return X_ml, X_test_ml, cat_cols

# -------------------------
# Logistic Regression ONLY
# -------------------------
def train_logreg(X_tr_sc, y_tr, X_val_sc, y_val):
    print("\n=== Training Logistic Regression ONLY ===")
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=500,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_tr_sc, y_tr)
    
    val_proba = model.predict_proba(X_val_sc)[:, 1]
    thr, f1 = best_threshold(y_val, val_proba)
    print(f"LogReg -> best_thr={thr:.3f}, F1={f1:.5f}")
    
    return model, val_proba, thr, f1

# -------------------------
# Main
# -------------------------
def main():
    # Load data
    print("Loading data...")
    X_raw, y, X_test_raw, test_ids, label_to_int, int_to_label = load_data()
    print(f"Train shape: {X_raw.shape}, Test shape: {X_test_raw.shape}")
    
    # Preprocess
    X_ml, X_test_ml, cat_cols = preprocess_for_models(X_raw, X_test_raw)
    print(f"Num categorical cols: {len(cat_cols)}, ML features: {X_ml.shape[1]}")
    
    # Global train/val split
    X_ml_tr, X_ml_val, y_tr, y_val = train_test_split(
        X_ml, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_ml_tr.shape[0]}, Val size: {X_ml_val.shape[0]}")
    
    # StandardScaler for Logistic Regression
    scaler = StandardScaler()
    X_ml_tr_sc = scaler.fit_transform(X_ml_tr)
    X_ml_val_sc = scaler.transform(X_ml_val)
    X_test_ml_sc = scaler.transform(X_test_ml)
    
    # Train ONLY Logistic Regression
    lr_model, lr_val_proba, lr_thr, lr_f1 = train_logreg(
        X_ml_tr_sc, y_tr, X_ml_val_sc, y_val
    )
    
    print(f"\n=== FINAL RESULTS: Logistic Regression ONLY ===")
    print(f"F1={lr_f1:.5f}, thr={lr_thr:.3f}")
    
    # Predict on test
    print("\nPredicting on test and writing submission...")
    p_final = lr_model.predict_proba(X_test_ml_sc)[:, 1]
    final_thr = lr_thr
    
    y_test_int = (p_final >= final_thr).astype(int)
    y_test_labels = pd.Series(y_test_int).map(int_to_label)
    
    out_name = f"{OUTPUT_CSV_PREFIX}_logreg_only_f1_{lr_f1:.4f}.csv"
    submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET_COL: y_test_labels,
    })
    submission.to_csv(out_name, index=False)
    
    print(f"\nSaved submission file: {out_name}")
    print(submission.head())

if __name__ == "__main__":
    main()
