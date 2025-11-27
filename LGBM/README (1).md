
# Unified ML Training Pipelines — XGBoost, CatBoost+SVM+NN Blend, LightGBM

This repository includes complete machine-learning pipelines used for founder retention prediction.  
Each pipeline applies strong feature engineering, leakage-safe techniques, and cross-validation to generate robust out-of-fold (OOF) metrics and final predictions.

---

# 1. XGBoost Pipeline (FE + Target Encoding + 5-Fold CV)

Source: `train_xgb_fe_te_kfold.py`

### Highlights
- Full feature engineering applied to train & test.
- Leakage-safe K-Fold Target Encoding.
- Label encoding for remaining categoricals.
- Stratified 5-Fold validation with early stopping.
- Saves:
  - `xgb_oof_proba.npy`
  - `xgb_test_proba.npy`
  - Submission CSV

### What It Produces
- Fold-wise metrics  
- OOF Accuracy & F1  
- Final test predictions  
- Probability files for ensembling

---

# 2. CatBoost + SVM + Neural Network Blended Pipeline

Source: `catboost_nn_svm.py`

### Highlights
- MAIN–HOLDOUT strategy:
  - MAIN (80%) → CatBoost 5-Fold
  - HOLDOUT (20%) → SVM + MLP
- Feature engineering identical across all models.
- Leakage-safe TE based only on MAIN.
- Separate numeric pipeline for SVM & NN.
- Blending:
  ```
  final = 0.8 * cat + 0.1 * svm + 0.1 * nn
  ```
- Saves:
  - `cat_main_oof_proba.npy`
  - `cat_main_test_proba.npy`
  - `svm_holdout.pkl`
  - `mlp_holdout.pkl`
  - `cat_final_on_main.pkl`
  - Final blended submission CSV

### What It Produces
- CatBoost MAIN OOF metrics  
- HOLDOUT evaluations (SVM, NN)  
- Blended probability and label outputs  

---

# 3. LightGBM FE K-Fold Pipeline

Source: `train_lgbm_fe_kfold.py`

### Highlights
- Uses same feature engineering as CatBoost pipeline.
- No target encoding; pure label encoding fold-wise.
- 5-Fold LightGBM training with best-iteration tracking.
- Saves:
  - `lgb_oof_proba.npy`
  - `lgb_test_proba.npy`
  - Submission CSV (threshold=0.5)

### What It Produces
- Fold-wise Accuracy & F1  
- OOF metrics  
- Final test predictions  

---

# 4. LightGBM Manual Tuning Script

Source: `tune_lgbm.py`

### Highlights
- Tests multiple strong LightGBM configurations.
- Automated evaluation of Accuracy, F1, and best iteration.
- Helps determine best params for final pipelines.

### What It Produces
- Sorted summary of tuning runs  
- Best-performing parameter set  

---

# Shared Components Across Pipelines

### ✔ Feature Engineering  
Includes engineered ratios, rating mappings, binnings, median fills, and ID removal.

### ✔ Leakage Prevention  
- TE only computed on proper folds / MAIN split.  
- Category encoders fitted safely.  
- HOLDOUT kept isolated where required.

### ✔ Probability Saving  
All major pipelines save OOF and test probabilities for later blending or threshold tuning.

---

# Notes

- All pipelines are compatible and designed for ensembling.
- Fully cross-validated with consistent preprocessing steps.
- Adaptable for Kaggle competitions, academic research, or production-style ML workflows.

