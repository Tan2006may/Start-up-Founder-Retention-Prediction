# K-Fold SVM Ensemble (LinearSVC)

This project builds a **5-fold SVM ensemble** to predict founder retention using stable feature engineering and threshold tuning.

## Pipeline Summary
1. Load `train.csv` and `test.csv`
2. Encode target with `LabelEncoder`  
3. Apply light, safe feature engineering  
4. Build preprocessing:
   - Numeric → median impute + StandardScaler  
   - Categorical → most_frequent + OneHotEncoder  
5. Train **5 LinearSVC models** using Stratified K-Fold  
6. Collect:
   - OOF decision scores (for threshold tuning)  
   - Test decision scores (averaged across folds)  
7. Sweep thresholds on OOF scores to find best F1  
8. Generate multiple submissions using tuned + shifted thresholds  
   - `svm_kfold_tuned_submission.csv`  
   - `svm_kfold_th*.csv`

## Why SVM Ensemble?
- LinearSVC is stable on high-dimensional OHE data  
- K-fold averaging reduces variance and improves generalization  
- OOF threshold tuning yields optimal classification behaviour

## Files Created
- `svm_kfold_tuned_submission.csv` (best threshold)  
- Several variants around the tuned threshold

  
## Summary
This script uses robust preprocessing, safe feature engineering, a 5-fold LinearSVC ensemble, OOF-based threshold tuning, and multiple threshold submissions to maximize F1-score on the leaderboard.

