import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def main():
    cat_test = np.load("cat_test_proba.npy")
    lgb_test = np.load("lgb_test_proba.npy")

    # simple mean blend
    blend_test = 0.5 * cat_test + 0.5 * lgb_test

    # default threshold first (safe)
    test_pred_labels = (blend_test >= 0.5).astype(int)

    # decode
    train = pd.read_csv("train.csv")
    le = LabelEncoder()
    le.fit(train["retention_status"])
    decoded = le.inverse_transform(test_pred_labels)

    sample = pd.read_csv("sample_submission.csv")
    sample[sample.columns[-1]] = decoded
    sample.to_csv("submission_blend_simple.csv", index=False)

    print("Saved submission_blend_simple.csv")

if __name__ == "__main__":
    main()
