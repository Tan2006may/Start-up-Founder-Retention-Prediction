# K-Fold SVM Ensemble (LinearSVC) for Retention Prediction

This project implements a robust 5-fold stratified ensemble using LinearSVC to predict retention status, with stable feature engineering and threshold tuning based on out-of-fold (OOF) decision scores. Multiple submission variants are generated for leaderboard optimization.

## Features
- Stable feature engineering creating interaction and ratio features such as stress_per_hour, revenue_per_member, funding_per_member, salary_revenue_ratio, workload_ratio, balance_gap, and stress_x_hours.
- Robust preprocessing pipeline with median imputation and StandardScaler for numeric features, and most frequent imputation plus One-Hot Encoding for categoricals.
- Uses StratifiedKFold (n_splits=5, shuffle=True, random_state=42) for balanced OOF splits.
- Trains LinearSVC with squared hinge loss, C=1.0, and max_iter=5000 for convergence.
- Decision function scores are used for precise threshold tuning on OOF predictions for optimal F1 score.
- Generates multiple test set submissions with thresholds around the tuned value for leaderboard tuning.

## Data
- Requires `train.csv` containing features and target column `retention_status`.
- Requires `test.csv` containing test features and `founder_id` for submission.


## Outputs
- `svm_kfold_tuned_submission.csv`: submission using best OOF-tuned threshold.
- Additional 5 submission CSVs: threshold varied ±0.2 and ±0.1 around the best threshold for leaderboard blending.
- Console displays per-fold progress, best OOF threshold with F1, and classification report.

## Hyperparameters
- LinearSVC: `C=1.0`, `loss="squared_hinge"`, `max_iter=5000`, `random_state=42`.
- StratifiedKFold: `n_splits=5`, `shuffle=True`, `random_state=42`.

## Notes
- Decision function scores are continuous, enabling more refined threshold selection than binary predictions.
- Feature engineering uses epsilons (+1) to avoid division by zero issues.
- Robust imputation ensures no missing values propagate.
- This approach provides a stable baseline suitable for stacking and ensemble methods.

