# Pure NN with Frequency Encoding (MLPClassifier)

Frequency-encoded categorical features + RobustScaler + class-weighted MLPClassifier. Manual hyperparameter search + threshold tuning for retention prediction.

## Features
- **Frequency encoding** for all categorical columns (no OneHot)
- Numeric feature engineering: stress_per_hour, revenue_per_member, funding_per_member, salary_revenue_ratio, workload_ratio, balance_gap, stress_x_hours, salary_x_stress
- **RobustScaler** + **class-weighted** MLPClassifier training
- Manual hyperparameter grid search over 4 architectures
- Threshold optimization on validation probabilities (0.1 - 0.9)
- Refits best model on full training data
**MLP params:** relu activation, adam solver, early_stopping, max_iter=200, random_state=42

## Workflow
1. Add numeric features to raw data.
2. Apply frequency encoding to categorical columns; drop originals.
3. Split encoded data into train/validation (80/20 stratified).
4. Compute class weights and apply as sample_weight in training.
5. Train MLPClassifier with each hyperparameter config.
6. For each config, tune classification threshold on validation probabilities (0.1 to 0.9) to maximize F1.
7. Select best config and threshold based on validation F1.
8. Refit best model on full training data with class weights.
9. Generate test predictions and save `nn_freq_final_submission.csv`.

## Usage
- Requires `train.csv` with `retention_status`.
- Requires `test.csv` with `founder_id`.

## Output
- `nn_freq_final_submission.csv`: Test predictions using tuned threshold from best model.
- Console logs include per-config validation F1 scores, best config summary, and classification report.

## Key Differences from SVM Ensemble
- Frequency encoding (compact categorical encoding) vs OneHot encoding.
- Non-linear MLPClassifier vs linear LinearSVC.
- Uses RobustScaler for numeric features.
- Handles class imbalance via sample weights.
- Threshold tuning on validation set probabilities vs decision_function scores.

This provides a strong pure neural network baseline using frequency encoding and adaptive thresholding.

