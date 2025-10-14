from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
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
        self.best_params = {}  # Store best hyperparameters

        if use_scaler:
            self.add_preprocessing('scaler', StandardScaler())

    def add_preprocessing(self, name, transformer):
        """Add a preprocessing step"""
        self.preprocessing_steps.append((name, transformer))
        return self

    def _get_fresh_preprocessors(self):
        """Get fresh copies of preprocessors to avoid state contamination"""
        return [(name, clone(transformer)) for name, transformer in self.preprocessing_steps]

    def _apply_preprocessing(self, X, preprocessors, fit=True):
        """Apply preprocessing steps using provided preprocessors"""
        X_transformed = X.copy() if isinstance(X, pd.DataFrame) else X.copy()
        for name, transformer in preprocessors:
            if fit:
                X_transformed = transformer.fit_transform(X_transformed)
            else:
                X_transformed = transformer.transform(X_transformed)
        return X_transformed

    def _cross_val_with_resampling(self, model, param_grid, X, y, sampler_name,
                                   scoring='f1_macro', cv=5, n_jobs=-1):
        """
        Performs cross-validation with proper preprocessing and resampling per fold.
        Returns: avg_metrics, mean_cv_score, best_params
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        fold_scores = []
        fold_metrics = []
        best_params_per_fold = []

        for fold_idx, (train_idx, val_idx) in enumerate(
                tqdm(skf.split(X, y), total=cv, desc="CV folds", leave=False)
        ):
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            # Check for single-class validation folds
            if len(np.unique(y_val_fold)) < 2:
                print(f"Warning: Fold {fold_idx} has only one class in validation set. Skipping.")
                continue

            # Get fresh preprocessors for this fold
            fold_preprocessors = self._get_fresh_preprocessors()

            # Fit preprocessing ONLY on training fold
            X_train_proc = self._apply_preprocessing(X_train_fold, fold_preprocessors, fit=True)
            X_val_proc = self._apply_preprocessing(X_val_fold, fold_preprocessors, fit=False)

            # Resample ONLY the training fold (after preprocessing)
            try:
                sampler_cls = RESAMPLERS[sampler_name]
                sampler = sampler_cls(random_state=self.random_state)
                X_resampled, y_resampled = sampler.fit_resample(X_train_proc, y_train_fold)
            except Exception as e:
                print(f"Resampling failed on fold {fold_idx}: {e}")
                continue

            # Inner grid search on resampled training data
            grid = GridSearchCV(model, param_grid, scoring=scoring, cv=3, n_jobs=n_jobs)
            grid.fit(X_resampled, y_resampled)
            best_model = grid.best_estimator_

            # Evaluate on UNTOUCHED validation fold
            y_pred = best_model.predict(X_val_proc)
            report = classification_report(y_val_fold, y_pred, output_dict=True, zero_division=0)

            # Store TRUE out-of-fold performance
            fold_score = report['macro avg']['f1-score']
            fold_scores.append(fold_score)
            fold_metrics.append({
                'f1_macro': report['macro avg']['f1-score'],
                'accuracy': report['accuracy'],
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall']
            })
            best_params_per_fold.append(grid.best_params_)

        # Average metrics across folds
        avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        mean_cv_score = np.mean(fold_scores)

        # Use most common best params (or from best fold)
        best_params = best_params_per_fold[np.argmax(fold_scores)] if best_params_per_fold else param_grid

        return avg_metrics, mean_cv_score, best_params

    def _retrain_on_full_data(self, model, best_params, X, y, sampler_name):
        """
        Retrain model on full training set with best hyperparameters.
        This is the model that should be used for final predictions.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Get fresh preprocessors and fit on full training set
        full_preprocessors = self._get_fresh_preprocessors()
        X_proc = self._apply_preprocessing(X, full_preprocessors, fit=True)

        # Resample full training set
        sampler_cls = RESAMPLERS[sampler_name]
        sampler = sampler_cls(random_state=self.random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_proc, y)

        # Train with best params
        final_model = clone(model).set_params(**best_params)
        final_model.fit(X_resampled, y_resampled)

        return final_model, full_preprocessors

    def random_forest(self, X, y, sampler, scoring='f1_macro', cv=5, n_jobs=-1):
        model = RandomForestClassifier(class_weight='balanced', random_state=self.random_state)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        return self._train_model(model, param_grid, X, y, sampler, 'rfc', scoring, cv, n_jobs)

    def xgboost(self, X, y, sampler, scoring='f1_macro', cv=5, n_jobs=-1):
        model = XGBClassifier(random_state=self.random_state, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'scale_pos_weight': [1, 3, 5]  # For class imbalance
        }
        return self._train_model(model, param_grid, X, y, sampler, 'xgb', scoring, cv, n_jobs)

    def mlp_classifier(self, X, y, sampler, scoring='f1_macro', cv=5, n_jobs=-1):
        model = MLPClassifier(early_stopping=True, random_state=self.random_state, max_iter=500)
        param_grid = {
            'hidden_layer_sizes': [(50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001],
            'learning_rate_init': [0.001, 0.01],
            'batch_size': [32, 64]
        }
        return self._train_model(model, param_grid, X, y, sampler, 'mlpc', scoring, cv, n_jobs)

    def _train_model(self, model, param_grid, X, y, sampler, label, scoring, cv, n_jobs):
        """
        Train one model with proper CV and retrain on full dataset.
        """
        print(f"\nTraining {label.upper()} with sampler: {sampler}")

        # Get CV metrics (preprocessing happens INSIDE CV)
        metrics, cv_score, best_params = self._cross_val_with_resampling(
            model, param_grid, X, y, sampler, scoring, cv, n_jobs
        )

        # Retrain on FULL training set with best params
        final_model, final_preprocessors = self._retrain_on_full_data(
            model, best_params, X, y, sampler
        )

        # Store everything
        model_key = f"{label}_{sampler}"
        self.trained_models[model_key] = (final_model, final_preprocessors)
        self.best_params[model_key] = best_params

        self.results.append({
            'model': model_key,
            'cv_score': cv_score,
            'best_params': best_params,
            **metrics
        })
        return self

    def fit_all(self, X, y, models=['rfc', 'xgb', 'mlpc'], samplers=None,
                scoring='f1_macro', cv=5, n_jobs=-1):
        """Train all specified models with all samplers"""
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
        """Evaluate all trained models on test set"""
        for i, result in enumerate(self.results):
            name = result['model']
            model, preprocessors = self.trained_models[name]

            # Use the model's own preprocessors
            X_test_proc = self._apply_preprocessing(X_test, preprocessors, fit=False)

            y_pred = model.predict(X_test_proc)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            # Update with test set performance
            self.results[i]['test_f1_macro'] = report['macro avg']['f1-score']
            self.results[i]['test_accuracy'] = report['accuracy']
            self.results[i]['test_precision'] = report['macro avg']['precision']
            self.results[i]['test_recall'] = report['macro avg']['recall']
        return self

    def get_results(self):
        """Return results as DataFrame"""
        return pd.DataFrame(self.results)

    def get_best_model(self, metric='cv_score'):
        """Get best model based on CV score (not test score to avoid leakage)"""
        df = self.get_results()
        best_idx = df[metric].idxmax()
        best_model_name = df.loc[best_idx, 'model']
        model, preprocessors = self.trained_models[best_model_name]
        return model, preprocessors, df.loc[best_idx]

    def predict(self, X, model_name=None):
        """Make predictions with preprocessing"""
        if model_name is None:
            model, preprocessors, _ = self.get_best_model()
        else:
            model, preprocessors = self.trained_models[model_name]

        X_proc = self._apply_preprocessing(X, preprocessors, fit=False)
        return model.predict(X_proc)

    def predict_proba(self, X, model_name=None):
        """Get prediction probabilities"""
        if model_name is None:
            model, preprocessors, _ = self.get_best_model()
        else:
            model, preprocessors = self.trained_models[model_name]

        X_proc = self._apply_preprocessing(X, preprocessors, fit=False)
        return model.predict_proba(X_proc)

    def add_custom_transform(self, name, transform_func, inverse_func=None):
        """Add custom preprocessing transformation"""

        class CustomTransform:
            def __init__(self, func, inv_func=None):
                self.func = func
                self.inv_func = inv_func

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return self.func(X)

            def fit_transform(self, X, y=None):
                return self.transform(X)

            def inverse_transform(self, X):
                if self.inv_func:
                    return self.inv_func(X)
                raise NotImplementedError("No inverse function provided")

        self.add_preprocessing(name, CustomTransform(transform_func, inverse_func))
        return self













































# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import (
#     SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
# )
# from imblearn.combine import SMOTEENN, SMOTETomek
#
# import pandas as pd
# from tqdm import tqdm
# import numpy as np
# import warnings
# warnings.filterwarnings('ignore')
#
# # Helper dictionary to access resampling methods by string
# RESAMPLERS = {
#     'smote': SMOTE,
#     'adasyn': ADASYN,
#     'borderline_smote': BorderlineSMOTE,
#     'svm_smote': SVMSMOTE,
#     'random_oversample': RandomOverSampler,
#     'smote_enn': SMOTEENN,
#     'smote_tomek': SMOTETomek
# }
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
#     def _cross_val_with_resampling(self, model, param_grid, X, y, sampler_name,
#                                    scoring='f1_macro', cv=5, n_jobs=-1):
#         """
#         Performs cross-validation with internal resampling.
#         - Applies preprocessing inside each fold (fit only on training fold)
#         - Avoids any data leakage
#         - Returns averaged metrics across folds
#         """
#
#         if isinstance(X, pd.DataFrame):
#             X = X.values
#         if isinstance(y, pd.Series):
#             y = y.values
#
#         skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
#         metrics = []
#         scores = []
#         best_models = []
#
#         for fold_idx, (train_idx, val_idx) in enumerate(
#                 tqdm(skf.split(X, y), total=cv, desc="CV folds", leave=False)
#         ):
#             X_train, X_val = X[train_idx], X[val_idx]
#             y_train, y_val = y[train_idx], y[val_idx]
#
#             # Check class presence to avoid single-class validation folds
#             if len(np.unique(y_val)) < 2:
#                 print(f"⚠️ Warning: Fold {fold_idx} has only one class in validation set.")
#                 continue
#
#             # Fit preprocessing ONLY on the training fold
#             X_train_proc = self._apply_preprocessing(X_train, fit=True)
#             X_val_proc = self._apply_preprocessing(X_val, fit=False)
#
#             # Fit resampler only on the preprocessed training fold
#             sampler_cls = RESAMPLERS[sampler_name]
#             sampler = sampler_cls()
#             X_res, y_res = sampler.fit_resample(X_train_proc, y_train)
#
#             # Inner grid search on the resampled training data
#             grid = GridSearchCV(model, param_grid, scoring=scoring, cv=3, n_jobs=n_jobs)
#             grid.fit(X_res, y_res)
#             best_model = grid.best_estimator_
#
#             # Evaluate on the untouched validation fold
#             y_pred = best_model.predict(X_val_proc)
#             report = classification_report(y_val, y_pred, output_dict=True)
#
#             # Collect per-fold metrics
#             metrics.append({
#                 'f1_macro': report['macro avg']['f1-score'],
#                 'accuracy': report['accuracy'],
#                 'precision': report['macro avg']['precision'],
#                 'recall': report['macro avg']['recall']
#             })
#             scores.append(report['macro avg']['f1-score'])  # true out-of-fold score
#             best_models.append(best_model)
#
#         # Average across folds
#         avg_metrics = pd.DataFrame(metrics).mean().to_dict()
#         best_model = best_models[np.argmax(scores)]
#         return avg_metrics, np.mean(scores), best_model
#
#
#
#     def random_forest(self, X, y, sampler, scoring='f1_macro', cv=5, n_jobs=-1):
#         model = RandomForestClassifier(class_weight='balanced', random_state=self.random_state)
#         param_grid = {
#             'n_estimators': [100, 200],
#             'max_depth': [10, 20],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2],
#             'max_features': ['sqrt']
#         }
#         return self._train_model(model, param_grid, X, y, sampler, 'rfc', scoring, cv, n_jobs)
#
#     def xgboost(self, X, y, sampler, scoring='f1_macro', cv=5, n_jobs=-1):
#         model = XGBClassifier(random_state=self.random_state, eval_metric='logloss')
#         param_grid = {
#             'n_estimators': [100, 200],
#             'max_depth': [3, 6],
#             'learning_rate': [0.01, 0.05],
#             'subsample': [0.7, 0.8],
#             'colsample_bytree': [0.7, 0.8]
#         }
#         return self._train_model(model, param_grid, X, y, sampler, 'xgb', scoring, cv, n_jobs)
#
#     def mlp_classifier(self, X, y, sampler, scoring='f1_macro', cv=5, n_jobs=-1):
#         model = MLPClassifier(early_stopping=True, random_state=self.random_state)
#         param_grid = {
#             'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
#             'activation': ['relu', 'tanh'],
#             'alpha': [0.0001, 0.001],
#             'learning_rate_init': [0.001, 0.01],
#             'learning_rate': ['constant', 'adaptive'],
#             'batch_size': [32, 64]
#         }
#         return self._train_model(model, param_grid, X, y, sampler, 'mlpc', scoring, cv, n_jobs)
#
#     def _train_model(self, model, param_grid, X, y, sampler, label, scoring, cv, n_jobs):
#         """
#         Trains one model type (e.g., RFC) with a given sampler using nested CV.
#         Preprocessing is handled *inside* CV to avoid leakage.
#         """
#         print(f"\nTraining {label.upper()} with sampler: {sampler}")
#
#         # No preprocessing here — that happens per fold inside cross-val
#         metrics, cv_score, best_model = self._cross_val_with_resampling(
#             model, param_grid, X, y, sampler, scoring, cv, n_jobs
#         )
#
#         # Save the best performing model and results
#         self.trained_models[f"{label}_{sampler}"] = best_model
#         self.results.append({
#             'model': f"{label}_{sampler}",
#             'cv_score': cv_score,  # mean F1 across folds
#             **metrics
#         })
#         return self
#
#     def fit_all(self, X, y, models=['rfc', 'xgb', 'mlpc'], samplers=None, scoring='f1_macro', cv=5, n_jobs=-1):
#         if samplers is None:
#             samplers = list(RESAMPLERS.keys())
#
#         for model_name in tqdm(models, desc="Models"):
#             for sampler in samplers:
#                 if model_name == 'rfc':
#                     self.random_forest(X, y, sampler, scoring, cv, n_jobs)
#                 elif model_name == 'xgb':
#                     self.xgboost(X, y, sampler, scoring, cv, n_jobs)
#                 elif model_name == 'mlpc':
#                     self.mlp_classifier(X, y, sampler, scoring, cv, n_jobs)
#                 else:
#                     print(f"Model {model_name} not recognized.")
#
#         return self
#
#     def evaluate(self, X_train, y_train, X_test, y_test):
#         self._apply_preprocessing(X_train, fit=True)
#         X_test_proc = self._apply_preprocessing(X_test, fit=False)
#
#         #X_test_proc = self._apply_preprocessing(X_test, fit=False)
#         for i, result in enumerate(self.results):
#             name = result['model']
#             model = self.trained_models[name]
#             y_pred = model.predict(X_test_proc)
#             report = classification_report(y_test, y_pred, output_dict=True)
#
#             self.results[i]['f1_macro'] = report['macro avg']['f1-score']
#             self.results[i]['accuracy'] = report['accuracy']
#             self.results[i]['precision'] = report['macro avg']['precision']
#             self.results[i]['recall'] = report['macro avg']['recall']
#         return self
#
#     def get_results(self):
#         return pd.DataFrame(self.results)
#
#     def get_best_model(self, metric='f1_macro'):
#         df = self.get_results()
#         best_idx = df[metric].idxmax()
#         best_model_name = df.loc[best_idx, 'model']
#         return self.trained_models[best_model_name], df.loc[best_idx]
#
#     def predict(self, X, model_name=None):
#         X_proc = self._apply_preprocessing(X, fit=False)
#         if model_name is None:
#             model, _ = self.get_best_model()
#         else:
#             model = self.trained_models[model_name]
#         return model.predict(X_proc)
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
#                 if self.inv_func:
#                     return self.inv_func(X)
#                 raise NotImplementedError("No inverse function provided")
#
#         self.add_preprocessing(name, CustomTransform(transform_func, inverse_func))
#         return self
#
#
#
#
#
#
