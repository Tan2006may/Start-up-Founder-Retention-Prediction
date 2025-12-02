import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score

# ===================================================
# LOAD DATA
# ===================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Convert string labels → numeric
y = train["retention_status"].map({"Left": 0, "Stayed": 1})
X = train.drop("retention_status", axis=1)

# Identify categorical + numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# ===================================================
# USE ONLY 20% OF THE TRAINING DATA
# ===================================================
X_20, _, y_20, _ = train_test_split(
    X, y,
    train_size=0.20,
    stratify=y,
    random_state=42
)

# ===================================================
# 80:20 SPLIT ON THE 20%
# ===================================================
X_tr, X_val, y_tr, y_val = train_test_split(
    X_20, y_20,
    test_size=0.20,
    stratify=y_20,
    random_state=42
)

# ===================================================
# PREPROCESSOR (IMPUTE → OHE → SCALE)
# ===================================================
preprocess = ColumnTransformer([
    ("cat", Pipeline([
        ("impute_cat", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols),

    ("num", Pipeline([
        ("impute_num", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), num_cols)
])

# ===================================================
# SVM MODEL
# ===================================================
svm_pipe = Pipeline([
    ("prep", preprocess),
    ("svm", CalibratedClassifierCV(
        LinearSVC(C=0.5, max_iter=5000),
        cv=3
    ))
])

svm_pipe.fit(X_tr, y_tr)
svm_pred = svm_pipe.predict(X_val)

svm_f1 = f1_score(y_val, svm_pred)
svm_acc = accuracy_score(y_val, svm_pred)

print("\n===== SVM RESULTS =====")
print("Accuracy:", round(svm_acc, 4))
print("F1 Score:", round(svm_f1, 4))

# ===================================================
# NEURAL NETWORK MODEL
# ===================================================
nn_pipe = Pipeline([
    ("prep", preprocess),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        learning_rate_init=0.001,
        max_iter=30,
        batch_size=64,
        random_state=42
    ))
])

nn_pipe.fit(X_tr, y_tr)
nn_pred = nn_pipe.predict(X_val)

nn_f1 = f1_score(y_val, nn_pred)
nn_acc = accuracy_score(y_val, nn_pred)

print("\n===== NEURAL NETWORK RESULTS =====")
print("Accuracy:", round(nn_acc, 4))
print("F1 Score:", round(nn_f1, 4))

# ===================================================
# CHOOSE THE BETTER MODEL AUTOMATICALLY
# ===================================================
print("\n==============================")
if svm_f1 >= nn_f1:
    best_model = svm_pipe
    best_name = "SVM"
    best_pred = svm_pipe.predict(test)
else:
    best_model = nn_pipe
    best_name = "Neural_Network"
    best_pred = nn_pipe.predict(test)

print(f"BEST MODEL SELECTED: {best_name}")
print("==============================")

# Convert predictions back to original labels
best_pred_text = pd.Series(best_pred).map({0: "Left", 1: "Stayed"})

submission = pd.DataFrame({
    "founder_id": test["founder_id"],
    "retention_status": best_pred_text
})

filename = f"{best_name}_20pct_submission.csv"
submission.to_csv(filename, index=False)

print(f"\nSaved final CSV as: {filename}")
