import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


RANDOM_STATE = 42
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
ID_COL = "founder_id"
TARGET_COL = "retention_status"
OUTPUT_CSV_PREFIX = "submission"  # final name will be built from best method & F1


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

    # ----- CatBoost data (raw cats + cleaned nums) -----
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for col in cat_cols:
        X_cb[col] = X_cb[col].astype(str).fillna("Missing")
        X_test_cb[col] = X_test_cb[col].astype(str).fillna("Missing")

    X_cb[num_cols], med_cb = clean_numeric(X_cb, num_cols)
    X_test_cb[num_cols], _ = clean_numeric(X_test_cb, num_cols, medians=med_cb)

    # ----- Numeric+freq for other models (XGB, LogReg, SVM, NN) -----
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
            X_test[col].astype(str).fillna("Missing"), freq_map
        )

    return X_cb, X_test_cb, X_ml, X_test_ml, cat_cols


# -------------------------
# Model trainers
# -------------------------

def train_catboost(X_tr_cb, y_tr, X_val_cb, y_val, cat_cols):
    cat_idx = [X_tr_cb.columns.get_loc(c) for c in cat_cols if c in X_tr_cb.columns]

    param_grid = [
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3},
        {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3},
        {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 5},
        {"depth": 10, "learning_rate": 0.03, "l2_leaf_reg": 5},
        {"depth": 10, "learning_rate": 0.02, "l2_leaf_reg": 7},
        {"depth": 12, "learning_rate": 0.02, "l2_leaf_reg": 7},
    ]

    best_model = None
    best_params = None
    best_thr = 0.5
    best_f1 = -1.0
    best_proba = None

    print("\n=== Tuning CatBoost ===")
    for i, params in enumerate(param_grid, start=1):
        print(f"\nCatBoost config {i}/{len(param_grid)}: {params}")
        model = CatBoostClassifier(
            random_seed=RANDOM_STATE,
            iterations=2000,
            early_stopping_rounds=100,
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params["l2_leaf_reg"],
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=200,
        )

        model.fit(
            X_tr_cb,
            y_tr,
            eval_set=(X_val_cb, y_val),
            cat_features=cat_idx,
            use_best_model=True,
        )

        val_proba = model.predict_proba(X_val_cb)[:, 1]
        thr, f1 = best_threshold(y_val, val_proba)
        print(f"   -> best_thr={thr:.3f}, F1={f1:.5f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_model = model
            best_params = params
            best_proba = val_proba

    print(f"\nBest CatBoost params: {best_params}, F1={best_f1:.5f}, thr={best_thr:.3f}")
    return best_model, best_proba, best_thr, best_f1


def train_xgb(X_tr_ml, y_tr, X_val_ml, y_val):
    param_grid = [
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 800},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 1000},
        {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 1500},
    ]

    best_model = None
    best_params = None
    best_thr = 0.5
    best_f1 = -1.0
    best_proba = None

    print("\n=== Tuning XGBoost ===")
    for i, params in enumerate(param_grid, start=1):
        print(f"\nXGB config {i}/{len(param_grid)}: {params}")
        model = XGBClassifier(
            random_state=RANDOM_STATE,
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            tree_method="hist",
        )

        model.fit(X_tr_ml, y_tr)
        val_proba = model.predict_proba(X_val_ml)[:, 1]
        thr, f1 = best_threshold(y_val, val_proba)
        print(f"   -> best_thr={thr:.3f}, F1={f1:.5f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_model = model
            best_params = params
            best_proba = val_proba

    print(f"\nBest XGB params: {best_params}, F1={best_f1:.5f}, thr={best_thr:.3f}")
    return best_model, best_proba, best_thr, best_f1


def train_logreg(X_tr_sc, y_tr, X_val_sc, y_val):
    print("\n=== Training Logistic Regression ===")
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


def train_svm_with_fractions(X_tr_sc, y_tr, X_val_sc, y_val, fractions=[0.2, 0.5]):
    print("\n=== Training SVM with different data fractions ===")
    best_model = None
    best_f1 = -1.0
    best_thr = 0.5
    best_proba = None
    best_frac = None

    for frac in fractions:
        print(f"\n--- SVM fraction = {frac:.2f} ---")
        X_sub, _, y_sub, _ = train_test_split(
            X_tr_sc, y_tr, train_size=frac, random_state=RANDOM_STATE, stratify=y_tr
        )
        model = SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
        )
        model.fit(X_sub, y_sub)
        val_proba = model.predict_proba(X_val_sc)[:, 1]
        thr, f1 = best_threshold(y_val, val_proba)
        print(f"   SVM(frac={frac:.2f}) -> best_thr={thr:.3f}, F1={f1:.5f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_proba = val_proba
            best_model = model
            best_frac = frac

    print(
        f"\nBest SVM: frac={best_frac:.2f}, F1={best_f1:.5f}, thr={best_thr:.3f}"
        if best_model is not None
        else "\nSVM did not train successfully."
    )
    return best_model, best_proba, best_thr, best_f1, best_frac


def train_nn_with_fractions(X_tr_sc, y_tr, X_val_sc, y_val, fractions=[0.2, 0.5, 1.0]):
    print("\n=== Training NN (MLP) with different data fractions ===")
    best_model = None
    best_f1 = -1.0
    best_thr = 0.5
    best_proba = None
    best_frac = None

    for frac in fractions:
        print(f"\n--- NN fraction = {frac:.2f} ---")

        if frac >= 1.0:
            # Use entire training dataset
            X_sub = X_tr_sc
            y_sub = y_tr
        else:
            X_sub, _, y_sub, _ = train_test_split(
                X_tr_sc, y_tr, train_size=frac, random_state=42, stratify=y_tr
            )

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=0.001,
            batch_size=256,
            max_iter=80,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10,
        )

        model.fit(X_sub, y_sub)

        val_proba = model.predict_proba(X_val_sc)[:, 1]
        thr, f1 = best_threshold(y_val, val_proba)
        print(f"   NN(frac={frac:.2f}) -> best_thr={thr:.3f}, F1={f1:.5f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_proba = val_proba
            best_model = model
            best_frac = frac

    print(f"\nBest NN: frac={best_frac}, F1={best_f1:.5f}, thr={best_thr:.3f}")
    return best_model, best_proba, best_thr, best_f1, best_frac


# -------------------------
# Ensemble
# -------------------------

def find_best_ensemble(y_val, model_probs_dict, top_k=3):
    """
    model_probs_dict: { name: {'proba': val_proba, 'f1': f1, 'thr': thr} }
    """
    # pick top_k by F1
    sorted_models = sorted(
        model_probs_dict.items(), key=lambda kv: kv[1]["f1"], reverse=True
    )
    top_models = sorted_models[:top_k]

    print("\nTop models for ensemble:")
    for name, info in top_models:
        print(f"  {name}: F1={info['f1']:.5f}")

    names = [name for name, _ in top_models]
    probs = [info["proba"] for _, info in top_models]

    weight_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_f1 = -1.0
    best_thr = 0.5
    best_weights = None

    print("\n=== Searching best ensemble weights ===")
    from itertools import product

    for ws in product(weight_vals, repeat=len(names)):
        if all(w == 0.0 for w in ws):
            continue
        w_sum = sum(ws)
        ws_norm = [w / w_sum for w in ws]
        p_ens = np.zeros_like(probs[0])
        for w, p in zip(ws_norm, probs):
            p_ens += w * p
        thr, f1 = best_threshold(y_val, p_ens)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_weights = ws_norm

    if best_weights is not None:
        print(
            f"Best ensemble over {names}: F1={best_f1:.5f}, thr={best_thr:.3f}, "
            f"weights={dict(zip(names, best_weights))}"
        )
    else:
        print("No valid ensemble weights found.")
    return names, best_weights, best_thr, best_f1


# -------------------------
# Main
# -------------------------

def main():
    # Load data
    print("Loading data...")
    X_raw, y, X_test_raw, test_ids, label_to_int, int_to_label = load_data()
    print(f"Train shape: {X_raw.shape}, Test shape: {X_test_raw.shape}")

    # Preprocess
    X_cb, X_test_cb, X_ml, X_test_ml, cat_cols = preprocess_for_models(X_raw, X_test_raw)
    print(f"Num categorical cols: {len(cat_cols)}, ML features: {X_ml.shape[1]}")

    # Global train/val split (will be used by all models)
    X_ml_tr, X_ml_val, y_tr, y_val = train_test_split(
        X_ml, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_cb_tr = X_cb.loc[X_ml_tr.index]
    X_cb_val = X_cb.loc[X_ml_val.index]

    print(f"Train size: {X_ml_tr.shape[0]}, Val size: {X_ml_val.shape[0]}")

    # StandardScaler for models that need scaling
    scaler = StandardScaler()
    X_ml_tr_sc = scaler.fit_transform(X_ml_tr)
    X_ml_val_sc = scaler.transform(X_ml_val)
    X_test_ml_sc = scaler.transform(X_test_ml)

    model_infos = {}  # name -> dict with model, proba, f1, thr, extras

    # ---- CatBoost ----
    cb_model, cb_val_proba, cb_thr, cb_f1 = train_catboost(
        X_cb_tr, y_tr, X_cb_val, y_val, cat_cols
    )
    model_infos["catboost"] = {
        "model": cb_model,
        "val_proba": cb_val_proba,
        "f1": cb_f1,
        "thr": cb_thr,
        "type": "cb",
    }

    # ---- XGBoost ----
    xgb_model, xgb_val_proba, xgb_thr, xgb_f1 = train_xgb(
        X_ml_tr, y_tr, X_ml_val, y_val
    )
    model_infos["xgb"] = {
        "model": xgb_model,
        "val_proba": xgb_val_proba,
        "f1": xgb_f1,
        "thr": xgb_thr,
        "type": "ml",
    }

    # ---- Logistic Regression ----
    lr_model, lr_val_proba, lr_thr, lr_f1 = train_logreg(
        X_ml_tr_sc, y_tr, X_ml_val_sc, y_val
    )
    model_infos["logreg"] = {
        "model": lr_model,
        "val_proba": lr_val_proba,
        "f1": lr_f1,
        "thr": lr_thr,
        "type": "ml_sc",
    }

    # ---- SVM with fractions ----
    svm_model, svm_val_proba, svm_thr, svm_f1, svm_frac = train_svm_with_fractions(
        X_ml_tr_sc, y_tr, X_ml_val_sc, y_val, fractions=[0.2, 0.5]
    )
    if svm_model is not None:
        model_infos["svm"] = {
            "model": svm_model,
            "val_proba": svm_val_proba,
            "f1": svm_f1,
            "thr": svm_thr,
            "type": "ml_sc",
            "frac": svm_frac,
        }

    # ---- NN (MLP) with fractions ----
    nn_model, nn_val_proba, nn_thr, nn_f1, nn_frac = train_nn_with_fractions(
        X_ml_tr_sc, y_tr, X_ml_val_sc, y_val, fractions=[0.2, 0.5, 1.0]
    )
    if nn_model is not None:
        model_infos["nn"] = {
            "model": nn_model,
            "val_proba": nn_val_proba,
            "f1": nn_f1,
            "thr": nn_thr,
            "type": "ml_sc",
            "frac": nn_frac,
        }

    # Summary single-model scores
    print("\n=== Single-model F1 scores on validation ===")
    for name, info in model_infos.items():
        print(f"{name}: F1={info['f1']:.5f}, thr={info['thr']:.3f}")

    # Prepare for ensemble: only models that have val_proba
    ensemble_dict = {
        name: {"proba": info["val_proba"], "f1": info["f1"], "thr": info["thr"]}
        for name, info in model_infos.items()
    }

    # Find best ensemble over top models
    ens_names, ens_weights, ens_thr, ens_f1 = find_best_ensemble(
        y_val, ensemble_dict, top_k=3
    )

    # Pick best strategy: best single vs ensemble
    best_single_name, best_single_info = max(
        model_infos.items(), key=lambda kv: kv[1]["f1"]
    )
    best_single_f1 = best_single_info["f1"]
    best_single_thr = best_single_info["thr"]

    print(
        f"\nBest single model: {best_single_name}, "
        f"F1={best_single_f1:.5f}, thr={best_single_thr:.3f}"
    )

    if ens_weights is not None and ens_f1 > best_single_f1:
        best_method = "ensemble"
        best_f1 = ens_f1
        best_thr = ens_thr
        print(
            f"\n=== Chosen method: ENSEMBLE over {ens_names} ===\n"
            f"F1={best_f1:.5f}, thr={best_thr:.3f}, weights={ens_weights}"
        )
    else:
        best_method = best_single_name
        best_f1 = best_single_f1
        best_thr = best_single_thr
        print(
            f"\n=== Chosen method: SINGLE MODEL '{best_method}' ===\n"
            f"F1={best_f1:.5f}, thr={best_thr:.3f}"
        )

    # -------------------------
    # Predict on test
    # -------------------------

    print("\nPredicting on test and writing submission...")

    # Precompute test probabilities for all models
    test_probas = {}

    for name, info in model_infos.items():
        mtype = info["type"]
        model = info["model"]

        if name == "catboost":
            p_test = model.predict_proba(X_test_cb)[:, 1]
        elif mtype == "ml":
            p_test = model.predict_proba(X_test_ml)[:, 1]
        elif mtype == "ml_sc":
            p_test = model.predict_proba(X_test_ml_sc)[:, 1]
        else:
            raise ValueError(f"Unknown model type for {name}: {mtype}")

        test_probas[name] = p_test

    if best_method == "ensemble":
        # combine top models with weights
        p_final = np.zeros_like(test_probas[ens_names[0]])
        for name, w in zip(ens_names, ens_weights):
            p_final += w * test_probas[name]
        final_thr = best_thr
    else:
        p_final = test_probas[best_method]
        final_thr = best_thr

    y_test_int = (p_final >= final_thr).astype(int)
    y_test_labels = pd.Series(y_test_int).map(int_to_label)

    out_name = f"{OUTPUT_CSV_PREFIX}_{best_method}_f1_{best_f1:.4f}.csv"
    submission = pd.DataFrame(
        {
            ID_COL: test_ids,
            TARGET_COL: y_test_labels,
        }
    )
    submission.to_csv(out_name, index=False)
    print(f"\nSaved submission file: {out_name}")
    print(submission.head())


if __name__ == "__main__":
    main()
