#-------Imports------
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

## models ##
from sklearn.neural_network import MLPClassifier #shallow NN
from sklearn.ensemble import RandomForestClassifier #tree based classifer
from xgboost import XGBClassifier #gradient boosted trees
## models ##

#random seed
random_seed = 42
np.random.seed(random_seed)

#-------Imports------
def train_test_split(df):
    target_col = 'target'
    feature_cols = ['civilian_targeting','fatalities','violence_against_women',
     'sub_event_type_Abduction/forced disappearance',
     'sub_event_type_Air/drone strike',
     'sub_event_type_Armed clash',
     'sub_event_type_Attack',
     'sub_event_type_Government regains territory',
     'sub_event_type_Grenade',
     'sub_event_type_Non-state actor overtakes territory',
     'sub_event_type_Remote explosive/landmine/IED',
     'sub_event_type_Sexual violence',
     'sub_event_type_Shelling/artillery/missile attack',
     'sub_event_type_Suicide bomb']

    X = working_df[feature_cols]
    y = working_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    return X_train, X_test, y_train, y_test

models = {
    #'mlpc': MLPClassifier(early_stopping=True, random_state=random_seed),
    'rfc': RandomForestClassifier(class_weight='balanced', random_state=random_seed),
    'xgb': XGBClassifier(random_state=random_seed, eval_metric='logloss')  # suppress warning
}

param_grids = {
#     'mlpc': {
#     'clf__hidden_layer_sizes': [(100,), (50, 50)],
#     'clf__activation': ['relu', 'tanh'],
#     'clf__solver': ['adam', 'sgd'],
#     'clf__alpha': [0.0001, 0.001],
#     'clf__learning_rate_init': [0.001, 0.01],
#     'clf__learning_rate': ['constant', 'adaptive'],
#     'clf__batch_size': [64, 128]
# },
    'rfc': { #random forrest
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20]
    },
    'xgb': { #gradient boosted trees
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 6],
        'clf__learning_rate': [0.05, 0.1]
    }
}

from tqdm import tqdm

results = []

for name, model in tqdm(models.items(), desc="traing models"):
    print(f"\nTraining model: {name.upper()}")

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])

    grid = GridSearchCV(pipe, param_grids[name], cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        'model': name,
        'best_params': grid.best_params_,
        'f1_macro': report['macro avg']['f1-score'],
        'accuracy': report['accuracy'],
        'estimator': best_model
    })

# === RESULTS === #
results_df = pd.DataFrame(results)
results_df

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Threshold for dropping features ===
importance_threshold = 0.001  # feel free to adjust this

# === Random Forest Feature Importance ===
rf_row = results_df[results_df['model'] == 'rfc'].iloc[0]
rf_model = rf_row['estimator']
rf_importances = rf_model.named_steps['clf'].feature_importances_

rf_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=rf_importance_df.head(45), x='importance', y='feature', palette='viridis')
plt.title('Top 45 Random Forest Feature Importances', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)

# Annotate bars
for i, v in enumerate(rf_importance_df.head(20)['importance']):
    plt.text(v + 0.001, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.show()

# === Drop suggestion from RF ===
rf_to_drop = rf_importance_df[rf_importance_df['importance'] < importance_threshold]
print(f"\n🔻 Features suggested to drop (Random Forest, importance < {importance_threshold}):")
print(rf_to_drop['feature'].tolist())



# === XGBoost Feature Importance ===
xgb_row = results_df[results_df['model'] == 'xgb'].iloc[0]
xgb_model = xgb_row['estimator']
xgb_importances = xgb_model.named_steps['clf'].feature_importances_

xgb_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=xgb_importance_df.head(45), x='importance', y='feature', palette='plasma')
plt.title('Top 45 XGBoost Feature Importances', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)

# Annotate bars
for i, v in enumerate(xgb_importance_df.head(20)['importance']):
    plt.text(v + 0.001, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.show()

# === Drop suggestion from XGB ===
xgb_to_drop = xgb_importance_df[xgb_importance_df['importance'] < importance_threshold]
print(f"\n🔻 Features suggested to drop (XGBoost, importance < {importance_threshold}):")
print(xgb_to_drop['feature'].tolist())