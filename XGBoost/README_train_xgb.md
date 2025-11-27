
# XGBoost Training Pipeline (Feature Engineering + Target Encoding + K-Fold)

This repository provides a complete training pipeline for a binary classification task using XGBoost. The workflow integrates structured feature engineering, leakage-safe target encoding, and stratified K-fold cross-validation to generate reliable out-of-fold (OOF) predictions and final test-set outputs.

---

## Feature Engineering

The pipeline enriches the dataset with additional numerical and categorical features, including:

- Ratio-based features:
  - `experience_ratio`
  - `funding_per_year`
  - `revenue_per_year`
  - `dependents_ratio`
  - `distance_adjusted`
- Ordinal encodings for rating-like columns:
  - work-life balance  
  - satisfaction  
  - performance  
  - reputation  
  - visibility
- Binning strategies applied via `cut` and `qcut` for:
  - founder age  
  - monthly revenue  
  - distance from hub  
  - years since founding
- Numeric type enforcement and median imputation  
- Cleanup of categorical inconsistencies  
- Removal of unique ID fields

These transformations help the model capture non-linear patterns and stabilize noisy signals.

---

## Target Encoding (Leakage-Safe)

A custom **Stratified K-Fold Target Encoding** mechanism is applied to selected categorical variables:

- `founder_role`  
- `education_background`  
- `personal_status`  
- `startup_stage`  
- `team_size_category`

Characteristics:

- For each fold, encoding is learned only from the training portion.  
- The global target mean is used for smoothing.  
- Test encodings are averaged across all folds.  

This ensures no leakage while improving representation of rare categories.

---

## Categorical Encoding

All non-target-encoded categorical features are label-encoded:

- Fitted on combined train + test data (safe, since target is not involved).
- Guarantees consistent category mappings across folds and for test samples.

---

## Model Training

The model uses an optimized XGBoost configuration:

- 2000 estimators  
- Learning rate: 0.05  
- Max depth: 6  
- Tree method: `hist`  
- `subsample = 0.8`, `colsample_bytree = 0.8`  
- Early stopping (100 rounds)  

Training is performed using **Stratified 5-Fold Cross-Validation**.  
For every fold, the script computes:

- Validation probabilities  
- OOF predictions  
- Fold-wise Accuracy and F1-score  
- Test predictions for ensemble averaging

---

## Evaluation

The script outputs:

- Overall OOF Accuracy  
- Overall OOF F1-Score  
- Classification report  
- Confusion matrix  

OOF predictions offer a realistic estimate of final model behavior.

---

## Final Outputs

The pipeline generates:

- A submission CSV (based on 0.5 threshold on averaged test probabilities)
- `xgb_oof_proba.npy` — stored OOF probabilities  
- `xgb_test_proba.npy` — averaged test probabilities  

These outputs support downstream tasks such as:

- Ensembling / model soups  
- Threshold optimization  
- ROC analysis  

---

## Notes

- Target leakage is fully prevented.  
- Pipeline is suitable for competitive ML workflows.  
- Easily extensible for improved feature engineering or tuning.
