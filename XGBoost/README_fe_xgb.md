
## What the script does
1. Loads `train.csv` and `test.csv`
2. Cleans data & performs extensive feature engineering  
3. Preprocesses using ColumnTransformer (scaling + OHE)
4. Selects top 120 features using RandomForest importance
5. Trains all major models on full data:
   - Logistic Regression  
   - KNN  
   - HistGradientBoosting  
   - AdaBoost  
   - RandomForest  
   - **XGBoost**  
   - LightGBM / CatBoost (if installed)
6. **Trains SVM and Neural Network on only 20% of data (project rule)**
7. Evaluates all models using F1-score
8. Picks best model (usually XGBoost)
9. Generates `submission_all_fe_xgb.csv`

## Why XGBoost wins
- Handles frequency-encoded categorical features extremely well  
- Learns non-linear interactions from engineered ratios, logs, and binning  
- Robust to missing values and mixed feature types  
- Uses **full dataset** (NN/SVM restricted to 20%)  
- Consistently achieves the highest F1-score


**Summary:**  
`model_all_fe_xgb.py` builds a fully engineered ML pipeline, compares many models, trains SVM/NN on a mandatory 20% subset, and finally uses **XGBoost** (best F1 performer) to generate predictions.

