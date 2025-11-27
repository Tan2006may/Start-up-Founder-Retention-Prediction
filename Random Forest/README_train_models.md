
# Multi-Model Training Script — LightGBM, XGBoost, Random Forest

This README describes the standalone training pipeline implemented in `train_models.py`.  
It is designed as a simple, fast, Windows-friendly all-in-one model trainer for founder retention prediction.

---

## Preprocessing

The script uses a clean and robust preprocessing strategy:

### Numeric Features  
- Filled using **median** of each column.

### Categorical Features  
- Filled using **mode**.
- Converted to string for consistency.
- Fitted with **LabelEncoder**.
- Unseen labels in test data are mapped to a new index (`len(classes)`).

This guarantees stable behavior across any dataset variation.

---

## Supported Models

The script trains one of three models based on the `--model` argument.

### 1. LightGBM
- 500 estimators  
- learning_rate = 0.05  
- num_leaves = 31  
- Early stopping (50 rounds)  
- Evaluation metric: binary logloss  
- Uses callbacks from modern LightGBM API  

### 2. XGBoost
- 400 estimators  
- learning_rate = 0.05  
- max_depth = 6  
- Tree method = hist  
- Early stopping = 50 rounds  
- eval_metric = logloss  

### 3. Random Forest
- 200 trees  
- Fully CPU-parallel (`n_jobs=-1`)  
- No early stopping  

---

## Training Procedure

1. Load `train.csv`, `test.csv`, `sample_submission.csv`
2. Separate features & target (`retention_status`)
3. Preprocess train → fit encoders  
4. Preprocess test → apply encoders  
5. Train/validation split: **80/20 stratified**
6. Train selected model (LGB/XGB/RF)
7. Evaluate:
   - Accuracy  
   - F1 score  
   - Classification report  
   - Confusion matrix  
8. Predict on test set  
9. Save:
   - Submission CSV  
   - Model pickle: `model_<model>.pkl`

---

## Output Files

- `submission.csv` (or custom name via `--output`)
- `model_lgb.pkl`, `model_xgb.pkl`, or `model_rf.pkl`

These files are production-ready and can be used for:
- Ensembling  
- Blend/soup models  
- Inference in other scripts  

---

## Notes

- No feature engineering is applied in this script (pure baseline).  
- Perfect for benchmarking different algorithms quickly.  
- Fully compatible with Windows & CPU-only training.  
- Encoders ensure unseen categories never crash the model.  

