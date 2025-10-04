import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from tqdm import tqdm


class ModelTrainer:
    """
    Custom class for training and evaluating multiple models with preprocessing.
    Each model has its own method with embedded grid search.
    """

    def __init__(self, random_state=42, use_scaler=True):
        self.random_state = random_state
        self.preprocessing_steps = []
        self.results = []
        self.trained_models = {}

        if use_scaler:
            self.add_preprocessing('scaler', StandardScaler())

    def add_preprocessing(self, name, transformer):
        """Add a preprocessing step (e.g., scaler, encoder, custom transform)"""
        self.preprocessing_steps.append((name, transformer))
        return self

    def _apply_preprocessing(self, X, fit=True):
        """Apply all preprocessing steps"""
        X_transformed = X.copy()
        for name, transformer in self.preprocessing_steps:
            if fit:
                X_transformed = transformer.fit_transform(X_transformed)
            else:
                X_transformed = transformer.transform(X_transformed)
        return X_transformed

    def random_forest(self, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1):
        """Train Random Forest with grid search"""
        print("\nTraining model: RANDOM FOREST")

        # Define model and param grid
        model = RandomForestClassifier(
            class_weight='balanced',
            random_state=self.random_state
        )
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }

        # Grid search
        grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid.fit(X_train, y_train)

        # Store results
        self.trained_models['rfc'] = grid.best_estimator_
        self.results.append({
            'model': 'rfc',
            'best_params': grid.best_params_,
            'cv_score': grid.best_score_,
            'estimator': grid.best_estimator_
        })

        return self

    def xgboost(self, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1):
        """Train XGBoost with grid search"""
        print("\nTraining model: XGBOOST")

        # Define model and param grid
        model = XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss'
        )
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.05, 0.1]
        }

        # Grid search
        grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid.fit(X_train, y_train)

        # Store results
        self.trained_models['xgb'] = grid.best_estimator_
        self.results.append({
            'model': 'xgb',
            'best_params': grid.best_params_,
            'cv_score': grid.best_score_,
            'estimator': grid.best_estimator_
        })

        return self

    def mlp_classifier(self, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1):
        """Train MLP Classifier with grid search"""
        print("\nTraining model: MLP CLASSIFIER")

        # Define model and param grid
        model = MLPClassifier(
            early_stopping=True,
            random_state=self.random_state
        )
        param_grid = {
            'hidden_layer_sizes': [(100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001],
            'learning_rate_init': [0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'batch_size': [64, 128]
        }

        # Grid search
        grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid.fit(X_train, y_train)

        # Store results
        self.trained_models['mlpc'] = grid.best_estimator_
        self.results.append({
            'model': 'mlpc',
            'best_params': grid.best_params_,
            'cv_score': grid.best_score_,
            'estimator': grid.best_estimator_
        })

        return self

    def fit_all(self, X_train, y_train, models=['rfc', 'xgb'], cv=5, scoring='f1_macro', n_jobs=-1):
        """Train all specified models"""
        # Apply preprocessing
        X_train_processed = self._apply_preprocessing(X_train, fit=True)

        # Train each model
        for model_name in tqdm(models, desc="Training models"):
            if model_name == 'rfc':
                self.random_forest(X_train_processed, y_train, cv, scoring, n_jobs)
            elif model_name == 'xgb':
                self.xgboost(X_train_processed, y_train, cv, scoring, n_jobs)
            elif model_name == 'mlpc':
                self.mlp_classifier(X_train_processed, y_train, cv, scoring, n_jobs)
            else:
                print(f"Warning: Model '{model_name}' not recognized")

        return self

    def evaluate(self, X_test, y_test):
        """Evaluate all trained models on test set"""
        X_test_processed = self._apply_preprocessing(X_test, fit=False)

        for i, result in enumerate(self.results):
            name = result['model']
            model = self.trained_models[name]

            # Predict
            y_pred = model.predict(X_test_processed)

            # Get metrics
            report = classification_report(y_test, y_pred, output_dict=True)

            # Update results
            self.results[i]['f1_macro'] = report['macro avg']['f1-score']
            self.results[i]['accuracy'] = report['accuracy']
            self.results[i]['precision'] = report['macro avg']['precision']
            self.results[i]['recall'] = report['macro avg']['recall']

        return self

    def get_results(self):
        """Return results as DataFrame"""
        return pd.DataFrame(self.results)

    def get_best_model(self, metric='f1_macro'):
        """Get the best performing model based on specified metric"""
        results_df = self.get_results()
        best_idx = results_df[metric].idxmax()
        best_model_name = results_df.loc[best_idx, 'model']
        return self.trained_models[best_model_name], results_df.loc[best_idx]

    def predict(self, X, model_name=None):
        """Make predictions using a specific model or the best model"""
        X_processed = self._apply_preprocessing(X, fit=False)

        if model_name is None:
            model, _ = self.get_best_model()
        else:
            model = self.trained_models[model_name]

        return model.predict(X_processed)

    def add_custom_transform(self, name, transform_func, inverse_func=None):
        """
        Add a custom transformation (for future encoder/decoder work)

        Args:
            name: Name of the transform
            transform_func: Function to apply transformation
            inverse_func: Optional function to reverse transformation
        """

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


# ============ USAGE EXAMPLE ============

# Initialize trainer
trainer = ModelTrainer(random_state=42)

# Add preprocessing
trainer.add_preprocessing('scaler', StandardScaler())

# Option 1: Train all models at once
# trainer.fit_all(X_train, y_train, models=['rfc', 'xgb'], cv=5, scoring='f1_macro', n_jobs=-1)

# Option 2: Train models individually
# X_train_processed = trainer._apply_preprocessing(X_train, fit=True)
# trainer.random_forest(X_train_processed, y_train)
# trainer.xgboost(X_train_processed, y_train)
# trainer.mlp_classifier(X_train_processed, y_train)

# Evaluate
# trainer.evaluate(X_test, y_test)

# Get results
# results_df = trainer.get_results()
# print(results_df)

# Get best model
# best_model, best_stats = trainer.get_best_model(metric='f1_macro')
# print(f"\nBest model: {best_stats['model']}")
# print(f"F1 Score: {best_stats['f1_macro']:.4f}")