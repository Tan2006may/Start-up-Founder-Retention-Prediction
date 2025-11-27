
# Submission Soup Blender — Probability & Label Ensembling

This README describes the submission‑blending utility implemented in `soup_binary.py`.  
It creates an averaged (or weighted) ensemble — commonly known as a **model soup** — from multiple submission CSV files.

The script automatically handles:
- Probability submissions with class columns  
- Label-only submissions  
- Alignment by `founder_id`  
- Weighted blending  
- Uniform fallback for unknown labels  

---

## What This Script Does

### ✔ Auto‑discovers submission CSVs  
If `--files` is not provided, the script scans the working directory and selects all CSVs except:
- `train.csv`
- `test.csv`
- `sample_submission.csv`
- The script itself
- Intermediate probability files (`oof`, `probas`, etc.)

### ✔ Reads each submission as probabilities  
Two supported formats:

1. **Probability submissions**  
   Columns match the class names from `train.csv`.

2. **Label submissions**  
   A single column named `retention_status`.  
   Converted into a one-hot probability matrix.

### ✔ Aligns rows by founder_id  
Regardless of order in each CSV, predictions are aligned to `test.csv` using the `founder_id` column.

### ✔ Weighted blending  
```
final = (w1 * p1 + w2 * p2 + ... + wn * pn) / sum(weights)
```

### ✔ Predicts final labels  
- Uses argmax over final probabilities  
- Maps class index back to label using `LabelEncoder` fitted on train labels  

### ✔ Saves final output  
Default output:  
```
submission_soup.csv
```

---

## How the Blending Works (Internals)

1. Load `train.csv` → extract class names using `LabelEncoder`
2. Load `test.csv` → preserve founder ordering
3. For each submission file:
   - Convert into probability matrix of shape `(n_test, n_classes)`
   - Reorder rows to match `test.csv`
4. Apply weights (default = equal)
5. Normalize rows
6. Compute final predictions
7. Save blended CSV with:
   - founder_id  
   - retention_status  

---

## Arguments

| Argument | Description |
|---------|-------------|
| `--files` | Explicit list of submission CSVs to blend |
| `--weights` | Corresponding weights (must match number of files) |
| `--out` | Output filename (default: `submission_soup.csv`) |
| `--verbose` | Prints alignment info & blending details |

Examples:
```
python soup_binary.py
python soup_binary.py --files s1.csv s2.csv
python soup_binary.py --files s1.csv s2.csv --weights 1 0.5
python soup_binary.py --files s1.csv s2.csv --weights 0.6 0.4 --out final.csv
```

---

## Notes

- Unknown labels in label submissions are converted to **uniform probabilities** (safe fallback).
- Rows are always sorted and aligned by founder_id before blending.
- Probability submissions with mismatched column order are automatically reordered.
- Supports unlimited number of submission files.

