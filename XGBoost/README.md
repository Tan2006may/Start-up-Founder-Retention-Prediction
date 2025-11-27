
This script performs:
- Data loading
- Cleaning & preprocessing
- Feature engineering
- Training **XGBoost on full data**
- Training **SVM and Neural Network (NN) on 20% data** (required rule)
- Evaluation using F1-score
- Threshold tuning
- Generating test predictions

---

## 2. Dataset Description
- `train.csv` — labelled training data  
- `test.csv` — unlabelled test data  

### Types of features:
- **Numerical features**: counts, activity metrics, business metrics  
- **Categorical features**: founder type, region, plan, industry  
- **Date features** (if present): converted to numeric offsets  

### Target:

---

## 3. Preprocessing
The script handles:
- Missing numerical values → median  
- Missing categorical values → mode  
- Type corrections for categorical columns  
- Removal or merging of rare labels (if needed)  
- Consistent train–test feature alignment  

---

## 4. Feature Engineering
The strongest part of the pipeline. Includes:

### 4.1 Frequency Encoding  
Used for **high-cardinality** categorical columns.  
Produces dense numeric features ideal for tree-based models (XGBoost).

### 4.2 One-Hot Encoding  
Used for **low-cardinality** categorical features.

### 4.3 Date Feature Extraction  
If dates exist in the dataset:
- month  
- day-of-week  
- days-difference  
- recency features  

### 4.4 Numerical Transformations  
- Median imputation  
- Optional log transforms  
- Scaling **only for NN/SVM**, not for XGB  

---

## 5. Models Trained in This Project

### 5.1 XGBoost (Final Selected Model)
Trained on **full training data**.

Why full data?
- XGBoost scales well  
- Handles all engineered features  
- Best generalization  
- Best F1-score  

Hyperparameters include:
- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`
- `colsample_bytree`
- Regularization parameters (`gamma`, `lambda`, etc.)

---

## 5.2 SVM (Support Vector Machine)
Per project rule:

> **SVM must be trained only on 20% of the dataset.**

Pipeline:
- 20% stratified train subset  
- Scaled numerical features  
- Frequency-encoded categorical features  

Limitations:
- High dimensionality  
- Hard to tune  
- Only 20% data → weaker learning  

Result:
- Lower F1 than XGBoost

---

## 5.3 Neural Network (NN)
Also trained on **20% data** as required.

Pipeline:
- StandardScaler  
- Dense MLP layers (hidden_sizes set in script)  
- Adam optimizer  

Limitations:
- Needs more data than allowed (but restricted to 20%)  
- Sensitive to hyperparameters  
- Underperforms XGB on this dataset  

Result:
- Lower F1 than XGBoost

---

## 6. Why XGBoost Became the Best Model

### ✔ Works perfectly with frequency-encoded features  
XGBoost learns splits over frequency values, capturing category importance better than SVM/NN.

### ✔ Handles non-linear interactions  
Retention behaviour depends on complex combinations of:
- region × activity  
- founder_type × subscription  
- business size × engagement  

XGB models these automatically.

### ✔ Robust to mixed feature types  
XGB doesn’t need standardized input and handles missing values internally.

### ✔ Uses the full dataset  
Unlike NN/SVM (restricted to 20%), XGB uses **100% of training data**, giving it a fundamental statistical advantage.

### ✔ Highest empirical F1-score  
From experiments:
- NN: lower  
- SVM: lower  
- LightGBM / RF: inconsistent  
- **XGBoost: best (~0.747 F1)**  
