# ============================================
#  SVM + NN ENSEMBLE WITH FEATURE ENGINEERING
#  - Auto FE (ratios, interactions, freq enc, ranks)
#  - RobustScaler
#  - SVM (LinearSVC + calibration) + MLP
#  - Threshold sweep near 0.74–0.78
#  - Multiple CSVs for leaderboard probing
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

# ============================================
#  LOAD DATA
# ============================================
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

TARGET = "retention_status"

y_raw = train_df[TARGET]
X_raw = train_df.drop(columns=[TARGET])

# Encode labels ('Left' / 'Stayed' -> 0/1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# ============================================
#  FEATURE ENGINEERING HELPERS
# ============================================

def add_basic_features(df):
    """Add ratios / interactions / gaps that are useful for retention-like tasks."""
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

    # Interactions
    if "stress_level" in df.columns and "hours_worked_per_week" in df.columns:
        df["stress_x_hours"] = df["stress_level"] * df["hours_worked_per_week"]

    if "salary" in df.columns and "stress_level" in df.columns:
        df["salary_x_stress"] = df["salary"] * df["stress_level"]

    if "revenue" in df.columns and "salary" in df.columns:
        df["revenue_div_salary"] = df["revenue"] / (df["salary"] + 1)

    if "funding_amount" in df.columns and "revenue" in df.columns:
        df["funding_div_revenue"] = df["funding_amount"] / (df["revenue"] + 1)

    if "stress_level" in df.columns and "work_life_balance_rating" in df.columns:
        df["balance_gap"] = df["stress_level"] - df["work_life_balance_rating"]

    return df


def add_frequency_encoding(X_train, X_test):
    """
    For each categorical column, add a *_freq feature = category relative frequency.
    No target leakage, uses only X distribution.
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        freqs = X_train[col].value_counts(normalize=True)
        X_train[col + "_freq"] = X_train[col].map(freqs).fillna(0)
        X_test[col + "_freq"]  = X_test[col].map(freqs).fillna(0)

    return X_train, X_test


def add_rank_features(X_train, X_test):
    """
    For each numeric column, add a *_rank feature = percentile rank (0–1).
    Computed separately in train and test (unsupervised, so safe).
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for col in num_cols:
        X_train[col + "_rank"] = X_train[col].rank(pct=True)
        if col in X_test.columns:
            X_test[col + "_rank"] = X_test[col].rank(pct=True)
        else:
            X_test[col + "_rank"] = 0.5

    return X_train, X_test


# ============================================
#  APPLY FEATURE ENGINEERING
# ============================================

# Start from copies of raw X and test
X = X_raw.copy()
test_features = test_df.copy()

# 1) Basic ratio / interaction features
X = add_basic_features(X)
test_features = add_basic_features(test_features)

# 2) Frequency encoding for all categorical columns
X, test_features = add_frequency_encoding(X, test_features)

# 3) Rank-based percentile features for numeric columns
X, test_features = add_rank_features(X, test_features)

# ============================================
#  PREPROCESSING (NUM + CAT) WITH RobustScaler
# ============================================
num_cols = X.select_dtypes(include=["int64", "float64", "float32"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# ============================================
#  TRAIN/VAL SPLIT
# ============================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================
#  DEFINE MODELS: SVM + NN
# ============================================

# SVM: LinearSVC + calibration
svm_base = LinearSVC(
    C=1.0,
    loss="squared_hinge",
    max_iter=5000,
    random_state=42
)
svm_clf = CalibratedClassifierCV(svm_base, cv=3)

# NN: MLP (slightly more regularized)
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=0.001,              # more regularization
    learning_rate="adaptive",
    max_iter=100,
    batch_size=64,
    random_state=42
)

# Ensemble: soft voting, SVM heavier weight
ensemble = VotingClassifier(
    estimators=[
        ("svm", svm_clf),
        ("nn", mlp_clf)
    ],
    voting="soft",
    weights=[3, 1]   # SVM dominates, NN adds nuance
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("ensemble", ensemble)
])

# ============================================
#  TRAIN ON TRAIN SPLIT
# ============================================
print("\nTraining SVM + NN ensemble with FE on train split...")
model.fit(X_train, y_train)

# ============================================
#  VALIDATION: THRESHOLD TUNING
# ============================================
probs_val = model.predict_proba(X_val)[:, 1]

best_f1 = -1
best_th = 0.5

for th in np.linspace(0.1, 0.9, 81):
    preds = (probs_val >= th).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print("\n============ ENSEMBLE + FE VAL RESULTS ============")
print(f"TUNED threshold: {best_th:.3f}  |  F1 (tuned): {best_f1:.4f}")

for fixed_th in [0.70, 0.74, 0.75, 0.76, 0.78]:
    preds_fixed = (probs_val >= fixed_th).astype(int)
    f1_fixed = f1_score(y_val, preds_fixed)
    acc_fixed = accuracy_score(y_val, preds_fixed)
    print(f"Threshold {fixed_th:.2f} -> F1: {f1_fixed:.4f} | Acc: {acc_fixed:.4f}")

print("\nClassification report at tuned threshold:\n")
preds_tuned = (probs_val >= best_th).astype(int)
print(classification_report(y_val, preds_tuned))
print("====================================================\n")

# ============================================
#  REFIT ON FULL TRAIN DATA
# ============================================
print("Refitting SVM + NN ensemble with FE on FULL training data...")
model.fit(X, y)

# ============================================
#  GENERATE MULTIPLE SUBMISSIONS AROUND 0.74–0.78
# ============================================
print("Generating predictions on test.csv for multiple thresholds...")

test_probs = model.predict_proba(test_features)[:, 1]

def make_and_save_submission(threshold, filename):
    preds_int = (test_probs >= threshold).astype(int)
    preds_labels = label_encoder.inverse_transform(preds_int)
    submission = pd.DataFrame({
        "founder_id": test_df["founder_id"],
        "retention_status": preds_labels
    })
    submission.to_csv(filename, index=False)
    print(f"Saved: {filename}  (threshold={threshold:.3f})")


# Keep tuned threshold for analysis
make_and_save_submission(best_th, "ensemble_fe_tuned_submission.csv")

# And a band around the LB sweet-spot
thresholds = [0.72, 0.74, 0.75, 0.76, 0.78]

for th in thresholds:
    fname = f"ensemble_fe_t{str(th).replace('.', 'p')}_submission.csv"
    make_and_save_submission(th, fname)

print("\nDone.\n")
