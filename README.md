# Start-up Founder Retention Prediction 

**Team Name:** Vector Space  
**Team Members:**  
- **Tanmay Kumar — IMT2023593**  
- **Yash Anand — IMT2023624**  
- **Rajan Samar — IMT2023616**  

**GitHub Repository:**  
https://github.com/Tan2006may/Start-up-Founder-Retention-Prediction

---

##  Project Overview

This project predicts whether a startup founder will be **retained (1)** or **not retained (0)**.  
The dataset is a large, structured tabular dataset containing mixed categorical and numerical features.

The workflow involves:

- Data cleaning and preprocessing  
- Feature engineering (Label/Freq/OHE Encoding)  
- Multiple ML models (Linear → Trees → Neural Nets)  
- Threshold tuning for F1 improvement  
- Final probability-level **model soup ensemble**  

**Final results:**

- **Best single-model (XGBoost) F1:** 0.747  
- **Final Model Soup ensemble F1:** **0.752**

---

##  Repository Structure


├── data/                       # Raw and processed data
├── XGBoost/                    # XGBoost training scripts & configs
├── LightGBM/                   # LightGBM experiments
├── CatBoost/                   # CatBoost experiments & ensemble files
├── Random Forest/              # Baseline Random Forest training
├── Logistic Regression/        # Logistic Regression implementation
├── SVM/                        # LinearSVC + calibrated OOF pipeline
├── Neural_Nets/                # MLP / Neural Network experiments
├── Ensemble Models/            # Soup / stacking scripts
├── requirements.txt            # Dependencies
└── README.md                   # This file




---

##  Preprocessing & Feature Engineering

###  Missing Values
- **Numerical:** median  
- **Categorical:** mode  

###  Encoding
- **Label Encoding:** high-cardinality categoricals  
- **Frequency Encoding:** selected categorical features  
- **One-Hot Encoding:** for LR, SVM, and NN  

###  Scaling
- **StandardScaler** applied for LR, SVM, and NN  
- Tree-based models use **unscaled** data  

###  Feature Selection
- Mutual information ranking  
- XGBoost feature importances  
- Final selected subsets: **Top-75 / Top-78** features  

###  Train/Validation Split
- **80/20 stratified split** on target label  

###  F1-Optimized Threshold Tuning
- Threshold sweep on validation probabilities  
- Crucial for LR, SVM, NN, XGBoost  

---

## Models & Performance Summary

| Model | Preprocessing | F1 (Validation) | Notes |
|-------|--------------|-----------------|-------|
| Logistic Regression | OHE + Scaling | ~0.65 | Simple linear baseline |
| SVM (LinearSVC + OOF) | OHE + Scaling | ~0.68–0.727 | Better than LR; still linear |
| Neural Network (MLP) | Dense Encode + Scaling | ~0.685–0.70 | Overfitting due to tabular data |
| Random Forest (200 trees) | Label Encode | Baseline | Simple tree baseline |
| LightGBM / CatBoost | Label/Freq Encode | Competitive | Slightly below XGBoost |
| **XGBoost (tuned)** | Label/Freq Encode + FS | **0.747** | Best single-model |
| **Model Soup Ensemble** | Probabilities | **0.752** | Best overall |

---

##  Final Ensemble — Model Soup

Rather than choosing a single model, we improved performance by averaging probability outputs from **three strong models**:

###  Files used in soup:
- `submission_all_fe_final.csv`  
- `submission_catboost_ensemble.csv`  
- `submission_catboost_ensemble_fe_K_fold_te.csv`  

###  How the Soup Works
1. Load all submission CSVs  
2. Convert predictions to probability matrices  
3. Align rows by `founder_id`  
4. Compute weighted average:
   P_final = (P1 + P2 + P3) / 3

###  Final Performance

Final predicted class = argmax of `P_final`
   F1 Score = 0.752

