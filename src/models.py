from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
)
from imblearn.combine import SMOTEENN, SMOTETomek

import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Helper dictionary to access resampling methods by string
RESAMPLERS = {
    'smote': SMOTE,
    'adasyn': ADASYN,
    'borderline_smote': BorderlineSMOTE,
    'svm_smote': SVMSMOTE,
    'random_oversample': RandomOverSampler,
    'smote_enn': SMOTEENN,
    'smote_tomek': SMOTETomek
}

class ModelTrainer:
    def __init__(self, random_state=42, use_scaler=True):
        self.random_state = random_state
        self.preprocessing_steps = []
        self.results = []
        self.trained_models = {}

        if use_scaler:
            self.add_preprocessing('scaler', StandardScaler())

    def add_preprocessing(self, name, transformer):
        self.preprocessing_steps.append((name, transformer))
        return self

    def _apply_preprocessing(self, X, fit=True):
        X_transformed = X.copy()
        for name, transformer in self.preprocessing_steps:
            if fit:
                X_transformed = transformer.fit_transform(X_transformed)
            else:
                X_transformed = transformer.transform(X_transformed)
        return X_transformed

    def _cross_val_with_resampling(self, model, param_grid, X, y, sampler_name, scoring='f1_macro', cv=5, n_jobs=-1):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = []
        metrics = []
        best_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=cv, desc="CV folds", leave=False)):
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            # --- FIX: Fit preprocessing only on training fold ---
            X_train_fold_proc = self._apply_preprocessing(X_train_fold, fit=True)
            X_val_fold_proc = self._apply_preprocessing(X_val_fold, fit=False)

            sampler_cls = RESAMPLERS[sampler_name]
            sampler = sampler_cls()
            X_resampled, y_resampled = sampler.fit_resample(X_train_fold_proc, y_train_fold)

            grid = GridSearchCV(model, param_grid, scoring=scoring, cv=3, n_jobs=n_jobs)
            grid.fit(X_resampled, y_resampled)

            best_model = grid.best_estimator_
            best_models.append(best_model)

            y_pred = best_model.predict(X_val_fold_proc)
            report = classification_report(y_val_fold, y_pred, output_dict=True)

            scores.append(grid.best_score_)
            metrics.append({
                'f1_macro': report['macro avg']['f1-score'],
                'accuracy': report['accuracy'],
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall']
            })

        avg_metrics = pd.DataFrame(metrics).mean().to_dict()
        return avg_metrics, np.mean(scores), best_models[np.argmax(scores)]

    # def _cross_val_with_resampling(self, model, param_grid, X, y, sampler_name, scoring='f1_macro', cv=5, n_jobs=-1):
    #     if isinstance(X, pd.DataFrame):
    #         X = X.values
    #     if isinstance(y, pd.Series):
    #         y = y.values
    #
    #     skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
    #     scores = []
    #     metrics = []
    #     best_models = []
    #
    #     for fold_idx, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=cv, desc="CV folds", leave=False)):
    #         X_train_fold, y_train_fold = X[train_idx], y[train_idx]
    #         X_val_fold, y_val_fold = X[val_idx], y[val_idx]
    #
    #         sampler_cls = RESAMPLERS[sampler_name]
    #         sampler = sampler_cls()
    #         X_resampled, y_resampled = sampler.fit_resample(X_train_fold, y_train_fold)
    #
    #         grid = GridSearchCV(model, param_grid, scoring=scoring, cv=3, n_jobs=n_jobs)
    #         grid.fit(X_resampled, y_resampled)
    #
    #         best_model = grid.best_estimator_
    #         best_models.append(best_model)
    #
    #         y_pred = best_model.predict(X_val_fold)
    #         report = classification_report(y_val_fold, y_pred, output_dict=True)
    #
    #         scores.append(grid.best_score_)
    #         metrics.append({
    #             'f1_macro': report['macro avg']['f1-score'],
    #             'accuracy': report['accuracy'],
    #             'precision': report['macro avg']['precision'],
    #             'recall': report['macro avg']['recall']
    #         })
    #
    #     avg_metrics = pd.DataFrame(metrics).mean().to_dict()
    #     return avg_metrics, np.mean(scores), best_models[np.argmax(scores)]

    def random_forest(self, X, y, sampler, scoring='f1_macro', cv=5, n_jobs=-1):
        model = RandomForestClassifier(class_weight='balanced', random_state=self.random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt']
        }
        return self._train_model(model, param_grid, X, y, sampler, 'rfc', scoring, cv, n_jobs)

    def xgboost(self, X, y, sampler, scoring='f1_macro', cv=5, n_jobs=-1):
        model = XGBClassifier(random_state=self.random_state, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8]
        }
        return self._train_model(model, param_grid, X, y, sampler, 'xgb', scoring, cv, n_jobs)

    def mlp_classifier(self, X, y, sampler, scoring='f1_macro', cv=5, n_jobs=-1):
        model = MLPClassifier(early_stopping=True, random_state=self.random_state)
        param_grid = {
            'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001],
            'learning_rate_init': [0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'batch_size': [32, 64]
        }
        return self._train_model(model, param_grid, X, y, sampler, 'mlpc', scoring, cv, n_jobs)

    def _train_model(self, model, param_grid, X, y, sampler, label, scoring, cv, n_jobs):
        print(f"\nTraining {label.upper()} with sampler: {sampler}")
        X_proc = self._apply_preprocessing(X, fit=True)
        metrics, cv_score, best_model = self._cross_val_with_resampling(model, param_grid, X_proc, y, sampler, scoring, cv, n_jobs)

        self.trained_models[f"{label}_{sampler}"] = best_model
        self.results.append({
            'model': f"{label}_{sampler}",
            'cv_score': cv_score,
            **metrics
        })
        return self

    def fit_all(self, X, y, models=['rfc', 'xgb', 'mlpc'], samplers=None, scoring='f1_macro', cv=5, n_jobs=-1):
        if samplers is None:
            samplers = list(RESAMPLERS.keys())

        for model_name in tqdm(models, desc="Models"):
            for sampler in samplers:
                if model_name == 'rfc':
                    self.random_forest(X, y, sampler, scoring, cv, n_jobs)
                elif model_name == 'xgb':
                    self.xgboost(X, y, sampler, scoring, cv, n_jobs)
                elif model_name == 'mlpc':
                    self.mlp_classifier(X, y, sampler, scoring, cv, n_jobs)
                else:
                    print(f"Model {model_name} not recognized.")

        return self

    def evaluate(self, X_test, y_test):
        X_test_proc = self._apply_preprocessing(X_test, fit=False)
        for i, result in enumerate(self.results):
            name = result['model']
            model = self.trained_models[name]
            y_pred = model.predict(X_test_proc)
            report = classification_report(y_test, y_pred, output_dict=True)

            self.results[i]['f1_macro'] = report['macro avg']['f1-score']
            self.results[i]['accuracy'] = report['accuracy']
            self.results[i]['precision'] = report['macro avg']['precision']
            self.results[i]['recall'] = report['macro avg']['recall']
        return self

    def get_results(self):
        return pd.DataFrame(self.results)

    def get_best_model(self, metric='f1_macro'):
        df = self.get_results()
        best_idx = df[metric].idxmax()
        best_model_name = df.loc[best_idx, 'model']
        return self.trained_models[best_model_name], df.loc[best_idx]

    def predict(self, X, model_name=None):
        X_proc = self._apply_preprocessing(X, fit=False)
        if model_name is None:
            model, _ = self.get_best_model()
        else:
            model = self.trained_models[model_name]
        return model.predict(X_proc)

    def add_custom_transform(self, name, transform_func, inverse_func=None):
        class CustomTransform:
            def __init__(self, func, inv_func=None):
                self.func = func
                self.inv_func = inv_func

            def fit(self, X, y=None): return self
            def transform(self, X): return self.func(X)
            def fit_transform(self, X, y=None): return self.transform(X)
            def inverse_transform(self, X):
                if self.inv_func:
                    return self.inv_func(X)
                raise NotImplementedError("No inverse function provided")

        self.add_preprocessing(name, CustomTransform(transform_func, inverse_func))
        return self





# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, f1_score
# from sklearn.model_selection import StratifiedKFold
# from tqdm import tqdm
#
# from imblearn.over_sampling import (
#     SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
# )
# from imblearn.combine import SMOTEENN, SMOTETomek
#
# # Global resampling methods (formerly in under_over.py)
# RESAMPLING_METHODS = {
#     'smote': SMOTE(random_state=42, k_neighbors=5),
#     'adasyn': ADASYN(random_state=42, n_neighbors=5),
#     'borderline_smote': BorderlineSMOTE(random_state=42, kind='borderline-1'),
#     'svm_smote': SVMSMOTE(random_state=42, k_neighbors=5),
#     'smote_enn': SMOTEENN(random_state=42),
#     'smote_tomek': SMOTETomek(random_state=42),
#     'random_oversample': RandomOverSampler(random_state=42)
# }
#
#
# def cross_val_with_resampling(model, X, y, sampler, cv=5, scoring='f1_macro'):
#     skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
#     scores = []
#
#     for train_idx, val_idx in skf.split(X, y):
#         X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
#         X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
#
#         X_res, y_res = sampler.fit_resample(X_train_fold, y_train_fold)
#         model.fit(X_res, y_res)
#         y_pred = model.predict(X_val_fold)
#
#         score = f1_score(y_val_fold, y_pred, average='macro')
#         scores.append(score)
#
#     return np.mean(scores)
#
#
# class ModelTrainer:
#     def __init__(self, random_state=42, use_scaler=True):
#         self.random_state = random_state
#         self.preprocessing_steps = []
#         self.results = []
#         self.trained_models = {}
#
#         if use_scaler:
#             self.add_preprocessing('scaler', StandardScaler())
#
#     def add_preprocessing(self, name, transformer):
#         self.preprocessing_steps.append((name, transformer))
#         return self
#
#     def _apply_preprocessing(self, X, fit=True):
#         X_transformed = X.copy()
#         for name, transformer in self.preprocessing_steps:
#             if fit:
#                 X_transformed = transformer.fit_transform(X_transformed)
#             else:
#                 X_transformed = transformer.transform(X_transformed)
#         return X_transformed
#
#     def fit_with_resampling(self, model_type, X_train, y_train, cv=5, scoring='f1_macro'):
#         for method_name, sampler in RESAMPLING_METHODS.items():
#             print(f"\n>>> Resampling with: {method_name.upper()} for {model_type.upper()}")
#
#             if model_type == 'rfc':
#                 model = RandomForestClassifier(random_state=self.random_state)
#             elif model_type == 'xgb':
#                 model = XGBClassifier(random_state=self.random_state, eval_metric='logloss')
#             elif model_type == 'mlpc':
#                 model = MLPClassifier(early_stopping=True, random_state=self.random_state)
#             else:
#                 print(f"Unknown model type: {model_type}")
#                 continue
#
#             try:
#                 X_train_processed = self._apply_preprocessing(X_train, fit=True)
#                 mean_cv_score = cross_val_with_resampling(model, pd.DataFrame(X_train_processed), y_train, sampler, cv, scoring)
#
#                 model.fit(*sampler.fit_resample(pd.DataFrame(X_train_processed), y_train))
#                 model_key = f"{model_type}_{method_name}"
#
#                 self.trained_models[model_key] = model
#                 self.results.append({
#                     'model': model_key,
#                     'oversample_method': method_name,
#                     'cv_score': mean_cv_score,
#                     'estimator': model
#                 })
#
#             except Exception as e:
#                 print(f"Failed on {method_name} for {model_type}: {e}")
#
#         return self
#
#     def fit_all(self, X_train, y_train, models=['rfc', 'xgb'], cv=5, scoring='f1_macro', n_jobs=-1):
#         for model_name in tqdm(models, desc="Training models"):
#             self.fit_with_resampling(model_name, X_train, y_train, cv=cv, scoring=scoring)
#         return self
#
#     def evaluate(self, X_test, y_test):
#         X_test_processed = self._apply_preprocessing(X_test, fit=False)
#
#         for i, result in enumerate(self.results):
#             name = result['model']
#             model = self.trained_models[name]
#
#             y_pred = model.predict(X_test_processed)
#             report = classification_report(y_test, y_pred, output_dict=True)
#
#             self.results[i]['f1_macro'] = report['macro avg']['f1-score']
#             self.results[i]['accuracy'] = report['accuracy']
#             self.results[i]['precision'] = report['macro avg']['precision']
#             self.results[i]['recall'] = report['macro avg']['recall']
#
#         return self
#
#     def get_results(self):
#         return pd.DataFrame(self.results)
#
#     def get_best_model(self, metric='f1_macro'):
#         results_df = self.get_results()
#         best_idx = results_df[metric].idxmax()
#         best_model_name = results_df.loc[best_idx, 'model']
#         return self.trained_models[best_model_name], results_df.loc[best_idx]
#
#     def predict(self, X, model_name=None):
#         X_processed = self._apply_preprocessing(X, fit=False)
#
#         if model_name is None:
#             model, _ = self.get_best_model()
#         else:
#             model = self.trained_models[model_name]
#
#         return model.predict(X_processed)
#
#     def add_custom_transform(self, name, transform_func, inverse_func=None):
#         class CustomTransform:
#             def __init__(self, func, inv_func=None):
#                 self.func = func
#                 self.inv_func = inv_func
#
#             def fit(self, X, y=None): return self
#             def transform(self, X): return self.func(X)
#             def fit_transform(self, X, y=None): return self.transform(X)
#             def inverse_transform(self, X):
#                 if self.inv_func: return self.inv_func(X)
#                 raise NotImplementedError("No inverse function provided")
#
#         self.add_preprocessing(name, CustomTransform(transform_func, inverse_func))
#         return self






# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from tqdm import tqdm
# from sklearn.model_selection import StratifiedKFold
# from imblearn.over_sampling import (
#     SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
# )
# from imblearn.combine import SMOTEENN, SMOTETomek
#
# class ModelTrainer:
#     """
#     Custom class for training and evaluating multiple models with preprocessing.
#     Each model has its own method with embedded grid search.
#     """
#
#     def __init__(self, random_state=42, use_scaler=True):
#         self.random_state = random_state
#         self.preprocessing_steps = []
#         self.results = []
#         self.trained_models = {}
#
#         if use_scaler:
#             self.add_preprocessing('scaler', StandardScaler())
#
#     def add_preprocessing(self, name, transformer):
#         """Add a preprocessing step (e.g., scaler, encoder, custom transform)"""
#         self.preprocessing_steps.append((name, transformer))
#         return self
#
#     def _apply_preprocessing(self, X, fit=True):
#         """Apply all preprocessing steps"""
#         X_transformed = X.copy()
#         for name, transformer in self.preprocessing_steps:
#             if fit:
#                 X_transformed = transformer.fit_transform(X_transformed)
#             else:
#                 X_transformed = transformer.transform(X_transformed)
#         return X_transformed
#
#     def random_forest(self, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1):
#         """Train Random Forest with grid search"""
#         print("\nTraining model: RANDOM FOREST")
#
#         # Define model and param grid
#         model = RandomForestClassifier(
#             class_weight='balanced',
#             random_state=self.random_state
#         )
#         param_grid = {
#             'n_estimators': [100, 200],
#             'max_depth': [None, 10, 20]
#         }
#
#         # Grid search
#         grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
#         grid.fit(X_train, y_train)
#
#         # Store results
#         self.trained_models['rfc'] = grid.best_estimator_
#         self.results.append({
#             'model': 'rfc',
#             'best_params': grid.best_params_,
#             'cv_score': grid.best_score_,
#             'estimator': grid.best_estimator_
#         })
#
#         return self
#
#     def xgboost(self, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1):
#         """Train XGBoost with grid search"""
#         print("\nTraining model: XGBOOST")
#
#         # Define model and param grid
#         model = XGBClassifier(
#             random_state=self.random_state,
#             eval_metric='logloss'
#         )
#         param_grid = {
#             'n_estimators': [100, 200],
#             'max_depth': [3, 6],
#             'learning_rate': [0.05, 0.1]
#         }
#
#         # Grid search
#         grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
#         grid.fit(X_train, y_train)
#
#         # Store results
#         self.trained_models['xgb'] = grid.best_estimator_
#         self.results.append({
#             'model': 'xgb',
#             'best_params': grid.best_params_,
#             'cv_score': grid.best_score_,
#             'estimator': grid.best_estimator_
#         })
#
#         return self
#
#     def mlp_classifier(self, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1):
#         """Train MLP Classifier with grid search"""
#         print("\nTraining model: MLP CLASSIFIER")
#
#         # Define model and param grid
#         model = MLPClassifier(
#             early_stopping=True,
#             random_state=self.random_state
#         )
#         param_grid = {
#             'hidden_layer_sizes': [(100,), (50, 50)],
#             'activation': ['relu', 'tanh'],
#             'solver': ['adam', 'sgd'],
#             'alpha': [0.0001, 0.001],
#             'learning_rate_init': [0.001, 0.01],
#             'learning_rate': ['constant', 'adaptive'],
#             'batch_size': [64, 128]
#         }
#
#         # Grid search
#         grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
#         grid.fit(X_train, y_train)
#
#         # Store results
#         self.trained_models['mlpc'] = grid.best_estimator_
#         self.results.append({
#             'model': 'mlpc',
#             'best_params': grid.best_params_,
#             'cv_score': grid.best_score_,
#             'estimator': grid.best_estimator_
#         })
#
#         return self
#
#     def fit_all(self, X_train, y_train, models=['rfc', 'xgb'], cv=5, scoring='f1_macro', n_jobs=-1):
#         """Train all specified models"""
#         # Apply preprocessing
#         X_train_processed = self._apply_preprocessing(X_train, fit=True)
#
#         # Train each model
#         for model_name in tqdm(models, desc="Training models"):
#             if model_name == 'rfc':
#                 self.random_forest(X_train_processed, y_train, cv, scoring, n_jobs)
#             elif model_name == 'xgb':
#                 self.xgboost(X_train_processed, y_train, cv, scoring, n_jobs)
#             elif model_name == 'mlpc':
#                 self.mlp_classifier(X_train_processed, y_train, cv, scoring, n_jobs)
#             else:
#                 print(f"Warning: Model '{model_name}' not recognized")
#
#         return self
#
#     def evaluate(self, X_test, y_test):
#         """Evaluate all trained models on test set"""
#         X_test_processed = self._apply_preprocessing(X_test, fit=False)
#
#         for i, result in enumerate(self.results):
#             name = result['model']
#             model = self.trained_models[name]
#
#             # Predict
#             y_pred = model.predict(X_test_processed)
#
#             # Get metrics
#             report = classification_report(y_test, y_pred, output_dict=True)
#
#             # Update results
#             self.results[i]['f1_macro'] = report['macro avg']['f1-score']
#             self.results[i]['accuracy'] = report['accuracy']
#             self.results[i]['precision'] = report['macro avg']['precision']
#             self.results[i]['recall'] = report['macro avg']['recall']
#
#         return self
#
#     def get_results(self):
#         """Return results as DataFrame"""
#         return pd.DataFrame(self.results)
#
#     def get_best_model(self, metric='f1_macro'):
#         """Get the best performing model based on specified metric"""
#         results_df = self.get_results()
#         best_idx = results_df[metric].idxmax()
#         best_model_name = results_df.loc[best_idx, 'model']
#         return self.trained_models[best_model_name], results_df.loc[best_idx]
#
#     def predict(self, X, model_name=None):
#         """Make predictions using a specific model or the best model"""
#         X_processed = self._apply_preprocessing(X, fit=False)
#
#         if model_name is None:
#             model, _ = self.get_best_model()
#         else:
#             model = self.trained_models[model_name]
#
#         return model.predict(X_processed)
#
#     def add_custom_transform(self, name, transform_func, inverse_func=None):
#         """
#         Add a custom transformation (for future encoder/decoder work)
#
#         Args:
#             name: Name of the transform
#             transform_func: Function to apply transformation
#             inverse_func: Optional function to reverse transformation
#         """
#
#         class CustomTransform:
#             def __init__(self, func, inv_func=None):
#                 self.func = func
#                 self.inv_func = inv_func
#
#             def fit(self, X, y=None):
#                 return self
#
#             def transform(self, X):
#                 return self.func(X)
#
#             def fit_transform(self, X, y=None):
#                 return self.transform(X)
#
#             def inverse_transform(self, X):
#                 if self.inv_func:
#                     return self.inv_func(X)
#                 raise NotImplementedError("No inverse function provided")
#
#         self.add_preprocessing(name, CustomTransform(transform_func, inverse_func))
#         return self
#
#
