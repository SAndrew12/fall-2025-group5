import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import (
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
    SVMSMOTE,
    RandomOverSampler
)
from imblearn.combine import SMOTEENN, SMOTETomek
import numpy as np

random_seed = 42
np.random.seed(random_seed)

# ---- Then undersample TRAIN ONLY ----
def undersample_train(X_train, y_train, majority_class=0, minority_class=1,
                      final_majority_size=1000, random_seed=42):
    rus = RandomUnderSampler(
        sampling_strategy={majority_class: final_majority_size,
                           minority_class: y_train.value_counts()[minority_class]},
        random_state=random_seed
    )
    X_train_und, y_train_und = rus.fit_resample(X_train, y_train)
    print("Class distribution after undersampling (train only):")
    print(y_train_und.value_counts())
    return X_train_und, y_train_und

# ----SMOTE ONLY----
def smote(X_train_in, y_train_in, random_seed=42, sampling_strategy='minority'):
    sm = SMOTE(random_state=random_seed, sampling_strategy=sampling_strategy)
    X_bal, y_bal = sm.fit_resample(X_train_in, y_train_in)
    print("Class distribution after SMOTE (train only):")
    print(y_bal.value_counts())
    return X_bal, y_bal
# ----SMOTE ONLY ----


#--- Test Different Oversample Teqs---
def ensamble_oversamp(X_train_in, y_train_in, random_seed=42,
                                sampling_strategy='minority'):
    """
    Generate all oversampled datasets using different techniques.

    Parameters:
    -----------
    X_train_in : array-like or DataFrame
        Training features (already undersampled)
    y_train_in : array-like or Series
        Training labels
    random_seed : int
        Random seed for reproducibility
    sampling_strategy : str or dict
        'minority' to balance classes, or custom dict

    Returns:
    --------
    dict : {method_name: (X_balanced, y_balanced)}
    """
    print("\n" + "=" * 60)
    print("CREATING ALL OVERSAMPLED DATASETS")
    print("=" * 60)

    # Initialize all oversampling methods
    methods = {
        'smote': SMOTE(random_state=random_seed, k_neighbors=5),
        'adasyn': ADASYN(random_state=random_seed, n_neighbors=5),
        'borderline_smote': BorderlineSMOTE(random_state=random_seed, kind='borderline-1'),
        'svm_smote': SVMSMOTE(random_state=random_seed, k_neighbors=5),
        'smote_enn': SMOTEENN(random_state=random_seed),
        'smote_tomek': SMOTETomek(random_state=random_seed),
        'random_oversample': RandomOverSampler(random_state=random_seed)
    }

    # Dictionary to store all oversampled datasets
    oversampled_data = {}

    # Generate each oversampled version
    for method_name, sampler in methods.items():
        try:
            print(f"\nGenerating: {method_name}...")
            sampler.sampling_strategy = sampling_strategy
            X_bal, y_bal = sampler.fit_resample(X_train_in, y_train_in)

            # Store the result
            oversampled_data[method_name] = (X_bal, y_bal)

            # Print class distribution
            unique, counts = np.unique(y_bal, return_counts=True)
            print(f"  Class distribution:")
            for cls, count in zip(unique, counts):
                print(f"    Class {cls}: {count}")

        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            continue

    print(f"\n✓ Successfully created {len(oversampled_data)} oversampled datasets")
    return oversampled_data
#---ensamble---


# --- Single Oversample Method (for flexibility) ---
def apply_oversample(X_train_in, y_train_in, method='smote', random_seed=42,
                     sampling_strategy='minority'):
    """
    Apply a single oversampling method.

    Parameters:
    -----------
    X_train_in : array-like or DataFrame
        Training features
    y_train_in : array-like or Series
        Training labels
    method : str
        Method to use: 'smote', 'adasyn', 'borderline_smote', 'svm_smote',
        'smote_enn', 'smote_tomek', 'random_oversample'
    random_seed : int
        Random seed
    sampling_strategy : str or dict
        Sampling strategy

    Returns:
    --------
    X_balanced, y_balanced
    """
    methods_dict = {
        'smote': SMOTE(random_state=random_seed, k_neighbors=5),
        'adasyn': ADASYN(random_state=random_seed, n_neighbors=5),
        'borderline_smote': BorderlineSMOTE(random_state=random_seed, kind='borderline-1'),
        'svm_smote': SVMSMOTE(random_state=random_seed, k_neighbors=5),
        'smote_enn': SMOTEENN(random_state=random_seed),
        'smote_tomek': SMOTETomek(random_state=random_seed),
        'random_oversample': RandomOverSampler(random_state=random_seed)
    }

    if method not in methods_dict:
        raise ValueError(f"Method '{method}' not recognized. Choose from: {list(methods_dict.keys())}")

    print(f"\nApplying {method}...")
    sampler = methods_dict[method]
    sampler.sampling_strategy = sampling_strategy
    X_bal, y_bal = sampler.fit_resample(X_train_in, y_train_in)

    print(f"Class distribution after {method}:")
    print(pd.Series(y_bal).value_counts())

    return X_bal, y_bal
#--- Single Oversample Method ---s---
