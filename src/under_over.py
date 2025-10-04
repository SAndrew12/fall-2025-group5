import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import numpy as np

random_seed = 42
np.random.seed(random_seed)

# ---- Then undersample TRAIN ONLY ----
def undersample_train(X_train, y_train, majority_class=0, minority_class=1,
                      final_majority_size=5000, random_seed=42):
    rus = RandomUnderSampler(
        sampling_strategy={majority_class: final_majority_size,
                           minority_class: y_train.value_counts()[minority_class]},
        random_state=random_seed
    )
    X_train_und, y_train_und = rus.fit_resample(X_train, y_train)
    print("Class distribution after undersampling (train only):")
    print(y_train_und.value_counts())
    return X_train_und, y_train_und

# ---- Optional: SMOTE once on the undersampled TRAIN ----
def smote(X_train_in, y_train_in, random_seed=42, sampling_strategy='minority'):
    sm = SMOTE(random_state=random_seed, sampling_strategy=sampling_strategy)
    X_bal, y_bal = sm.fit_resample(X_train_in, y_train_in)
    print("Class distribution after SMOTE (train only):")
    print(y_bal.value_counts())
    return X_bal, y_bal