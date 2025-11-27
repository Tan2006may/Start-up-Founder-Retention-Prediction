# Logistic Regression ONLY (Frequency Encoded)

## Features
- **Frequency encoding** for all categorical columns
- **Median imputation** + inf handling for numerics
- **StandardScaler** normalization
- **Class weight balanced** training
- Automatic **F1 threshold tuning** on validation set
- 80/20 stratified train/validation split


**Requires:** `train.csv` (with `retention_status`), `test.csv` (with `founder_id`)

## Output
- `submission_logreg_only_f1_X.XXXX.csv`: Tuned threshold predictions
- Console: Val F1 score, optimal threshold, submission preview
