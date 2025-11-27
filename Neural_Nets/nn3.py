# ============================================
#  PURE NN MODEL WITH FREQ ENCODING (NO JOBLIB)
#  - No SVM anywhere
#  - Categorical -> frequency encoded (no OHE)
#  - Numeric FE + interactions
#  - RobustScaler + MLPClassifier
#  - Class-weighted training
#  - Manual hyperparam + threshold search
#  - nn_freq_final_submission.csv
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

TARGET = "retention_status"

y_raw = train_df[TARGET]
X_raw = train_df.drop(columns=[TARGET])

# encode target ('Left'/'Stayed' -> 0/1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# -----------------------------------------
# BASIC NUMERIC FEATURE ENGINEERING
# -----------------------------------------
def add_numeric_features(df):
    df = df.copy()

    if "stress_level" in df.columns and "hours_worked_per_week" in df.columns:
        df["stress_per_hour"] = df["stress_level"] / (df["hours_worked_per_week"] + 1)

    if "revenue" in df.columns and "team_size" in df.columns:
        df["revenue_per_member"] = df["revenue"] / (df["team_size"] + 1)

    if "funding_amount" in df.columns and "team_size" in df.columns:
        df["funding_per_member"] = df["funding_amount"] / (df["team_size"] + 1)

    if "salary" in df.columns and "revenue" in df.columns:
        df["salary_revenue_ratio"] = df["salary"] / (df["revenue"] + 1)

    if "hours_worked_per_week" in df.columns and "work_life_balance_rating" in df.columns:
        df["workload_ratio"] = df["hours_worked_per_week"] / (df["work_life_balance_rating"] + 1)

    # interactions
    if "stress_level" in df.columns and "work_life_balance_rating" in df.columns:
        df["balance_gap"] = df["stress_level"] - df["work_life_balance_rating"]

    if "stress_level" in df.columns and "hours_worked_per_week" in df.columns:
        df["stress_x_hours"] = df["stress_level"] * df["hours_worked_per_week"]

    if "salary" in df.columns and "stress_level" in df.columns:
        df["salary_x_stress"] = df["salary"] * df["stress_level"]

    return df

X_fe    = add_numeric_features(X_raw)
test_fe = add_numeric_features(test_df)

# -----------------------------------------
# FREQUENCY ENCODING FOR CATEGORICAL COLUMNS
# -----------------------------------------
def freq_encode(train_df, test_df):
    train = train_df.copy()
    test  = test_df.copy()

    cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        freqs = train[col].value_counts(normalize=True)
        train[col + "_freq"] = train[col].map(freqs).fillna(0)
        test[col + "_freq"]  = test[col].map(freqs).fillna(0)

    # Drop original categorical columns (keep only *_freq)
    train = train.drop(columns=cat_cols)
    test  = test.drop(columns=cat_cols)

    return train, test

X_enc, test_enc = freq_encode(X_fe, test_fe)

# -----------------------------------------
# PREPROCESSING (NUMERIC ONLY NOW)
# -----------------------------------------
num_cols = X_enc.columns.tolist()  # everything is numeric after freq-encoding

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ]), num_cols)
    ]
)

# -----------------------------------------
# TRAIN / VAL SPLIT
# -----------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_enc, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------------------
# CLASS WEIGHTS -> SAMPLE WEIGHTS
# -----------------------------------------
class_counts = np.bincount(y_train)
total = len(y_train)
class_weights = {cls: total / (2.0 * cnt) for cls, cnt in enumerate(class_counts)}
sample_weight_train = np.array([class_weights[c] for c in y_train])

# -----------------------------------------
# MANUAL HYPERPARAM SEARCH FOR PURE NN
# -----------------------------------------
configs = [
    {"hidden_layer_sizes": (256, 128, 64), "alpha": 1e-4, "lr": 3e-4, "batch_size": 128},
    {"hidden_layer_sizes": (256, 128),     "alpha": 5e-4, "lr": 3e-4, "batch_size": 128},
    {"hidden_layer_sizes": (128, 64),      "alpha": 1e-4, "lr": 1e-3, "batch_size": 64},
    {"hidden_layer_sizes": (256, 256, 128),"alpha": 5e-5, "lr": 3e-4, "batch_size": 128},
]

best_overall_f1 = -1.0
best_overall_th = 0.5
best_config = None
best_model = None

print("\n=== Manual hyperparam search for PURE NN (freq-encoded) ===")

for idx, cfg in enumerate(configs, start=1):
    print(f"\n--- Config {idx}/{len(configs)} ---")
    print(cfg)

    mlp = MLPClassifier(
        hidden_layer_sizes=cfg["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=cfg["alpha"],
        learning_rate="adaptive",
        learning_rate_init=cfg["lr"],
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        batch_size=cfg["batch_size"],
        random_state=42
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("mlp", mlp)
    ])

    model.fit(X_train, y_train, mlp__sample_weight=sample_weight_train)

    # probs on val
    probs_val = model.predict_proba(X_val)[:, 1]

    # threshold search
    best_f1_cfg = -1.0
    best_th_cfg = 0.5
    for th in np.linspace(0.1, 0.9, 81):
        preds = (probs_val >= th).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1_cfg:
            best_f1_cfg = f1
            best_th_cfg = th

    print(f"Config {idx} best F1 on val: {best_f1_cfg:.4f} at threshold {best_th_cfg:.3f}")

    if best_f1_cfg > best_overall_f1:
        best_overall_f1 = best_f1_cfg
        best_overall_th = best_th_cfg
        best_config = cfg
        best_model = model

print("\n==== BEST CONFIG (PURE NN) ====")
print(best_config)
print(f"Val F1 (best config): {best_overall_f1:.4f} at threshold {best_overall_th:.3f}")

# final val metrics for best model
probs_val_best = best_model.predict_proba(X_val)[:, 1]
val_preds_best = (probs_val_best >= best_overall_th).astype(int)
print("\nClassification report (val, best NN):\n")
print(classification_report(y_val, val_preds_best))
print("=========================================\n")

# -----------------------------------------
# REFIT BEST NN ON FULL TRAIN DATA (WITH WEIGHTS)
# -----------------------------------------
print("Refitting BEST NN on FULL training data (freq-encoded)...")

# recompute weights on full y
class_counts_full = np.bincount(y)
total_full = len(y)
class_weights_full = {cls: total_full / (2.0 * cnt) for cls, cnt in enumerate(class_counts_full)}
sample_weight_full = np.array([class_weights_full[c] for c in y])

mlp_full = MLPClassifier(
    hidden_layer_sizes=best_config["hidden_layer_sizes"],
    activation="relu",
    solver="adam",
    alpha=best_config["alpha"],
    learning_rate="adaptive",
    learning_rate_init=best_config["lr"],
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    batch_size=best_config["batch_size"],
    random_state=42
)

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("mlp", mlp_full)
])

final_model.fit(X_enc, y, mlp__sample_weight=sample_weight_full)

# -----------------------------------------
# FINAL PREDICTIONS ON TEST + CSV
# -----------------------------------------
print("Generating predictions on test.csv with PURE NN (freq-encoded)...")

test_probs = final_model.predict_proba(test_enc)[:, 1]
test_preds_int = (test_probs >= best_overall_th).astype(int)
test_preds_labels = label_encoder.inverse_transform(test_preds_int)

submission = pd.DataFrame({
    "founder_id": test_df["founder_id"],
    "retention_status": test_preds_labels
})

submission.to_csv("nn_freq_final_submission.csv", index=False)

print("\nSaved: nn_freq_final_submission.csv\n")
