import pandas as pd
from data_loader import load_data
from feature_eng import feature_creating
from train_test_split import t_t_s
from under_over import undersample_train
from under_over import smote
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
X_train_und, y_train_und = undersample_train(X_train, y_train)
print(X_train_und.shape)
print(y_train_und.shape)
#---Under Sample---






#---Over Sample---
X_bal, y_bal = smote(X_train_und, y_train_und)
print(X_bal.shape)
print(y_bal.shape)

#---Over Sample---





###---CLASSICAL MODELS---###
trainer = ModelTrainer()

# Train both RF and XGBoost models
trainer.fit_all(X_bal, y_bal, models=['rfc', 'xgb'], cv=5, scoring='f1_macro', n_jobs=-1)

# Evaluate on test set
trainer.evaluate(X_test, y_test)

# Get and display results
results_df = trainer.get_results()
print("\n" + "="*60)
print("MODEL RESULTS")
print("="*60)
print(results_df[['model', 'cv_score', 'f1_macro', 'accuracy', 'precision', 'recall']])

# Get best model
best_model, best_stats = trainer.get_best_model(metric='f1_macro')
print("\n" + "="*60)
print(f"BEST MODEL: {best_stats['model'].upper()}")
print("="*60)
print(f"Test F1 Score: {best_stats['f1_macro']:.4f}")
print(f"Test Accuracy: {best_stats['accuracy']:.4f}")
print(f"CV Score: {best_stats['cv_score']:.4f}")


###---CLASSICAL MODELS---###


