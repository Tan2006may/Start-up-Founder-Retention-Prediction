
This script performs:
- Data loading  
- Preprocessing  
- Feature engineering  
- Training an XGBoost model  
- Validation and F1 scoring  
- Threshold tuning  
- Test-set inference  
- Saving the final predictions  

---

## 1. File: `model_fe_xgb.py`
The script implements the complete ML pipeline:

### **✔ Data Loading**
Reads:
- `train.csv`
- `test.csv`

Ensures:
- correct dtypes  
- categorical → string  
- missing value handling  

---

## 2. Feature Engineering
This code applies **strong feature engineering**, which is why XGBoost performs best.

### **Included techniques:**
#### **2.1 Frequency Encoding**
High-cardinality categorical features are converted into frequency/count features.  
This is essential because:
- XGBoost performs best with dense numeric inputs  
- One-hot creates extremely sparse high-dimensional matrices  
- Frequency encoding captures category importance without explosion of dimensions

#### **2.2 One-Hot Encoding**  
Used only for low-cardinality columns.

#### **2.3 Date Features**
The script extracts:
- day  
- month  
- difference between dates  
- activity duration  
- recency features  

These dramatically help tree models.

#### **2.4 Numerical Transformations**
- missing value imputation  
- log transforms (if applied in script)  
- scaling (only for diagnostics, XGB uses raw values)  

---

## 3. Model: XGBoost (XGBClassifier)
The code trains XGBoost using tuned hyperparameters such as:
- `learning_rate`
- `max_depth`
- `subsample`
- `colsample_bytree`
- `n_estimators`
- `scale_pos_weight` (if used)
- `gamma`, `min_child_weight`

The model is trained on the engineered dataset and evaluated using:
- **Validation F1 score**
- **Threshold-tuned decision boundary** (important for imbalanced data)

---

## 4. Why XGBoost is the Best Model
XGBoost consistently becomes the top-performing model **because the feature engineering used in this project aligns perfectly with how XGBoost learns**.

### **XGBoost strengths that match this project’s data:**

#### **✔ Works extremely well with frequency-encoded categorical features**
Logistic Regression and NN cannot exploit frequency-encoded categorical values well.  
XGB uses splits on these values → stronger decision boundaries.

#### **✔ Naturally handles non-linear interactions**
The dataset includes:
- behavioural patterns  
- time-based relationships  
- industry/region interactions  

Only gradient-boosted trees can model these automatically.

#### **✔ Tolerant to missing values**
XGB learns optimal “missing” directions during tree growth.

#### **✔ Does not require scaling**
Mixed-scale features (counts, amounts, encoded categories) work directly.

#### **✔ High interpretability via feature importance**
The code outputs importance plots / values showing which engineered features matter.

### **Empirical reason (from your experiments):**
Your pipeline showed:
- Logistic Regression: ~0.74  
- LightGBM: sometimes ~0.74  
- CatBoost: inconsistent + slower  
- NN/SVM: unstable & low F1  
- **XGBoost: ~0.747 (BEST)**

This is because your engineered features (freq encoding + date deltas + aggregated stats)
match XGBoost’s boosting structure perfectly.

**Hence XGBoost generalizes better and gives the highest F1 score.**

---

## 5. Running the Script

### **Install dependencies**
