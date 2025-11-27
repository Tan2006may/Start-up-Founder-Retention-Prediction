# Project Overview: ensemble1.py & model__nn2.py

This repository contains two Python scripts implementing different machine learning approaches for retention status prediction.

---

## ensemble1.py

- Implements multiple classical and modern models including Logistic Regression, SVM, XGBoost, CatBoost, and MLPClassifier.
- Preprocesses data with frequency encoding for categorical variables and median imputation for numerics.
- Searches for optimal classification thresholds per model using F1 score on validation data.
- Trains and validates models on stratified splits, including experiments with different data fractions for SVM and NN.
- Combines predictions via weighted ensemble with a grid search over ensemble weights to optimize validation F1.
- Outputs predictions from best single model or weighted ensemble on the test set.
- Provides a comprehensive training pipeline supporting automated thresholding, ensemble selection, and submission generation.

---

## model__nn2.py

- Focuses on a pure neural network approach using MLPClassifier.
- Applies feature engineering including numeric interactions and uses One-Hot Encoding for categorical features.
- Uses RobustScaler and median imputation for preprocessing.
- Conducts hyperparameter tuning via RandomizedSearchCV over hidden layers, alpha, learning rates, and batch sizes.
- Applies early stopping and validation for model selection.
- Tunes classification threshold on validation probabilities to maximize F1 score.
- Re-trains the best model on the full dataset with refined hyperparameters.
- Generates final predictions saved as `nn_final_submission.csv`.

---

# Summary

- `ensemble1.py` is a versatile script combining multiple models with frequency-encoded features and ensemble learning to maximize predictive performance.
- `model__nn2.py` specializes in tuning and deploying a single MLP neural network model with One-Hot Encoding and rigorous hyperparameter search.
- Both workflows include threshold tuning for improved classification thresholds and output submission files for leaderboard evaluation.
