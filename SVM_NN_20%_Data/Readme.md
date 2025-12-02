#  SVM vs Neural Network on 20% Data

This experiment evaluates how a **Linear SVM** and a **Neural Network (MLP)** perform when trained on only **20% of the dataset**, simulating a low-data scenario.

##  Preprocessing
Both models use the same preprocessing pipeline:
- Numerical → Median imputation + StandardScaler  
- Categorical → Mode imputation + One-Hot Encoding  
- Labels mapped: `Left → 0`, `Stayed → 1`

##  Data Split
- 20% of full training data sampled (stratified)  
- Inside that: **80% train, 20% validation**  
- Ensures identical and fair input to both models

##  Models
### **Linear SVM**
- `LinearSVC(C=0.5)` with probability calibration  
- Strong on sparse OHE tabular data

### **Neural Network (MLP)**
- Architecture: `(128, 64)` with ReLU  
- LR = 0.001, batch = 64, epochs = 30  

## Results
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **SVM** | 0.7493 | **0.7618** |
| MLP | 0.7103 | 0.7335 |

## Final Selection
The script automatically selects the model with the highest F1-score.  
Here, **SVM performs better** and is chosen for final test predictions.
