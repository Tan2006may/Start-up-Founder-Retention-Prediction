# ============================================
#  K-FOLD SVM ENSEMBLE (LinearSVC)
#  - Simple FE (same as stable one)
#  - Robust preprocessing
#  - 5-fold OOF scores for threshold tuning
#  - Test scores averaged across folds
#  - Threshold sweep 0.74–0.78
#  - Multiple CSVs for leaderboard
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

TARGET = "retention_status"

y_raw = train_df[TARGET]
X_raw = train_df.drop(columns=[TARGET])

# encode labels ('Left'/'Stayed' -> 0/1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# -----------------------------------------
# SIMPLE, STABLE FEATURE ENGINEERING
# (this is the version that did not hurt LB)
# -----------------------------------------
def feature_engineer(df):
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

    # a couple of interactions that are unlikely to hurt
    if "stress_level" in df.columns and "work_life_balance_rating" in df.columns:
        df["balance_gap"] = df["stress_level"] - df["work_life_balance_rating"]

    if "stress_level" in df.columns and "hours_worked_per_week" in df.columns:
        df["stress_x_hours"] = df["stress_level"] * df["hours_worked_per_week"]

    return df

X = feature_engineer(X_raw)
test_features = feature_engineer(test_df)

# -----------------------------------------
# PREPROCESSING
# -----------------------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
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

# -----------------------------------------
# K-FOLD SVM ENSEMBLE
# -----------------------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_scores = np.zeros(len(X))
test_scores_sum = np.zeros(len(test_features))

fold_idx = 1

for train_index, val_index in kf.split(X, y):
    print(f"\n=== Fold {fold_idx} / 5 ===")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # base SVM
    svm = LinearSVC(
        C=1.0,
        loss="squared_hinge",
        max_iter=5000,
        random_state=42
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("svm", svm)
    ])

    model.fit(X_train, y_train)

    # decision_function gives continuous scores (can be negative/positive)
    val_scores = model.decision_function(X_val)
    oof_scores[val_index] = val_scores

    # scores on test set
    test_fold_scores = model.decision_function(test_features)
    test_scores_sum += test_fold_scores

    fold_idx += 1

# average test scores across folds
test_scores_avg = test_scores_sum / kf.n_splits

# -----------------------------------------
# TUNE THRESHOLD ON OOF SCORES
# -----------------------------------------
print("\n=== Threshold tuning on OOF scores ===")
score_min, score_max = oof_scores.min(), oof_scores.max()

best_f1 = -1.0
best_th = 0.0

for th in np.linspace(score_min, score_max, 300):
    preds = (oof_scores >= th).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"Best OOF threshold: {best_th:.4f} | OOF F1: {best_f1:.4f}")

# just for sanity: report metrics at that threshold
oof_preds = (oof_scores >= best_th).astype(int)
print("\nClassification report on OOF (using tuned threshold):\n")
print(classification_report(y, oof_preds))

# -----------------------------------------
# GENERATE SUBMISSIONS AROUND 0.75–0.78 REGION
# BUT NOW IN *SCORE SPACE*
# -----------------------------------------

print("\nGenerating submissions from K-fold SVM ensemble...")

def make_and_save_submission_from_scores(th, filename):
    preds_int = (test_scores_avg >= th).astype(int)
    preds_labels = label_encoder.inverse_transform(preds_int)
    submission = pd.DataFrame({
        "founder_id": test_df["founder_id"],
        "retention_status": preds_labels
    })
    submission.to_csv(filename, index=False)
    print(f"Saved: {filename}  (score threshold={th:.4f})")


# 1) use the globally tuned threshold
make_and_save_submission_from_scores(best_th, "svm_kfold_tuned_submission.csv")

# 2) tweak around the tuned threshold, but also give you 5 variants
offsets = [-0.2, -0.1, 0.0, 0.1, 0.2]  # in score space

for off in offsets:
    th = best_th + off
    th_str = f"{th:.4f}".replace('.', 'p')   # only modify the numeric part
    fname = f"svm_kfold_th{th_str}.csv"      # keep proper .csv extension
    make_and_save_submission_from_scores(th, fname)

print("\nDone.\n")
