import pandas as pd
from data_loader import load_data
from feature_eng import feature_creating
from train_test_split import t_t_s
from under_over import undersample_train
from under_over import smote
from under_over import ensamble_oversamp
from models import ModelTrainer




###---LOAD DATA---###
df = load_data()
###---LOAD DATA---###

###---PRE-PROCESS + FEATURES---###
working_df, unattrib_df  = feature_creating(df)
print(working_df.shape)
print(working_df['target'].value_counts())
print(working_df['target'].unique())
###---PRE-PROCESS + FEATURES---###

#---Train Test Split---
X_train, X_test, y_train, y_test = t_t_s(working_df)

print(X_train.shape)
print(y_train.shape)
#---Train Test Split---



#---Under Sample---
X_train_und, y_train_und = undersample_train(X_train, y_train, final_majority_size=2500)
print(X_train_und.shape)
print(y_train_und.shape)
#---Under Sample---






#---SMOTE ONLY---
# X_bal, y_bal = smote(X_train_und, y_train_und)
# print(X_bal.shape)
# print(y_bal.shape)
#---SMOTE ONLY---

#---All Oversample---
oversampled_datasets = ensamble_oversamp(X_train_und, y_train_und,
                                         random_seed=42,
                                         sampling_strategy='minority')
#---All Oversample---





###---CLASSICAL MODELS---###
all_results = []

for method_name, (X_bal, y_bal) in oversampled_datasets.items():
    print("\n" + "=" * 80)
    print(f"TRAINING MODELS WITH {method_name.upper()} OVERSAMPLING")
    print("=" * 80)
    print(f"Dataset shape: {X_bal.shape}")
    print(f"Class distribution: {pd.Series(y_bal).value_counts().to_dict()}")

    # Initialize a new trainer for each oversampling method
    trainer = ModelTrainer()

    # Train both RF and XGBoost models
    trainer.fit_all(X_bal, y_bal, models=['rfc', 'xgb', 'mlpc'], cv=5, scoring='f1_macro', n_jobs=-1)

    # Evaluate on test set
    trainer.evaluate(X_test, y_test)

    # Get results and add method name
    results_df = trainer.get_results()
    results_df['oversample_method'] = method_name

    # Store results
    all_results.append(results_df)

    # Display results for this method
    print(f"\nResults for {method_name}:")
    print(results_df[['model', 'cv_score', 'f1_macro', 'accuracy', 'precision', 'recall']])

###---TRAIN MODELS ON EACH OVERSAMPLED DATASET---###


###---COMPARE ALL RESULTS---###
# Combine all results
combined_results = pd.concat(all_results, ignore_index=True)

print("\n" + "=" * 80)
print("COMPLETE RESULTS COMPARISON")
print("=" * 80)
print(combined_results[['oversample_method', 'model', 'cv_score', 'f1_macro',
                        'accuracy', 'precision', 'recall']].to_string())

# Find best overall combination
best_idx = combined_results['f1_macro'].idxmax()
best_result = combined_results.loc[best_idx]

print("\n" + "=" * 80)
print("BEST OVERALL COMBINATION")
print("=" * 80)
print(f"Oversampling Method: {best_result['oversample_method'].upper()}")
print(f"Model: {best_result['model'].upper()}")
print(f"Test F1 Score: {best_result['f1_macro']:.4f}")
print(f"Test Accuracy: {best_result['accuracy']:.4f}")
print(f"CV Score: {best_result['cv_score']:.4f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall: {best_result['recall']:.4f}")

# Save results to CSV for later analysis
combined_results.to_csv('oversample_comparison_results.csv', index=False)
print("\nResults saved to 'oversample_comparison_results.csv'")
###---COMPARE ALL RESULTS---###

###---CLASSICAL MODELS---###


