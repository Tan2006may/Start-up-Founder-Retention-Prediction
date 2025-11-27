
# CatBoost + SVM + Neural Network Blended Training Pipeline

This repository contains a full machine-learning workflow that blends CatBoost, SVM, and Neural Network (MLP) models for binary classification. The script performs feature engineering, leakage-safe target encoding, holdout-based training for secondary models, and weighted blending to generate final predictions.

---

## Feature Engineering

The pipeline enhances the dataset using:

- Numeric cleaning and median imputation  
- Ratio-based engineered features:
  - `experience_ratio`
  - `funding_per_year`
  - `revenue_per_year`
  - `dependents_ratio`
  - `distance_adjusted`
- Ordinal mappings for:
  - work-life balance
  - satisfaction
  - performance
  - reputation
  - visibility
- Additional interaction features:
  - `stress_factor`
  - `satisfaction_x_visibility`
  - `performance_x_reputation`
- Removal of identifier columns
- Consistent transformation across train, holdout, and test data

These features help capture hidden relationships and improve generalization across models.

---

## Target Encoding (Main-Only, Leakage-Free)

Target encoding is performed **only on the MAIN split (80%)** using stratified K‑Fold:

- Columns encoded:
  - `founder_role`
  - `education_background`
  - `personal_status`
  - `startup_stage`
  - `team_size_category`
- Test-set encodings are averaged across folds
- HOLDOUT (20%) encodings use full-main statistics (no leakage)
- Smoothing applied with global target mean

This setup ensures the holdout and test data never see future information.

---

## Model Training Strategy

### 1. CatBoost (Primary Model — trained on MAIN only)

- Trained using 5‑Fold stratified cross-validation  
- Computes OOF predictions and fold-wise test probabilities  
- CatBoost settings:
  - depth: 6  
  - learning rate: 0.05  
  - 1000 iterations with early stopping  
  - F1 as evaluation metric  
- Final CatBoost model is also trained on full MAIN and saved as a pickle  
- Outputs:
  - `cat_main_oof_proba.npy`
  - `cat_main_test_proba.npy`
  - `cat_final_on_main.pkl`

---

### 2. SVM (Trained on HOLDOUT Only)

- HOLDOUT (20%) data is processed into fully numeric matrix (LabelEncoding + scaling + imputation)
- Uses RBF kernel with probability calibration
- Evaluated on HOLDOUT for sanity check
- Model saved as:
  - `svm_holdout.pkl`

---

### 3. Neural Network (MLP — trained on HOLDOUT)

- Architecture: (128 → 64) fully connected  
- Max iterations: 500  
- Trained on HOLDOUT numeric matrix  
- Evaluated on HOLDOUT  
- Model saved as:
  - `mlp_holdout.pkl`

---

## Blending

Final test probabilities are computed as:

```
final_proba = w_cat * catboost + w_svm * svm + w_nn * mlp
```

Default weights:

- CatBoost: **0.8**
- SVM: **0.1**
- Neural Network: **0.1**

Threshold is applied to convert probabilities → final labels (default = 0.5).

Outputs saved:

- `blended_test_proba.npy`
- `blended_oof_cat_on_main_proba.npy`
- Final submission CSV

---

## Evaluation Summary

- CatBoost OOF metrics (MAIN split) are printed:
  - F1 score
  - Accuracy
  - Confusion matrix  
- HOLDOUT evaluations for SVM and MLP are shown for verification  
- Ensures the blend incorporates both strong MAIN-learned structure and HOLDOUT‑learned patterns

---

## Notes

- Pipeline is fully leakage‑safe (target encoding always isolated).  
- SVM and MLP never access MAIN, ensuring unbiased behavior.  
- Easily adjustable blend weights, threshold, and architecture.  
- Suitable for competitions, research, and ensemble‑based ML pipelines.

