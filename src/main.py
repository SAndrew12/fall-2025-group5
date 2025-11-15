import pandas as pd
import numpy as np
import torch
from data_loader import load_data
from feature_eng import feature_creating
from feature_eng import mask_group_names, mask_location_names
from train_test_split import t_t_s
from under_over import undersample_train
from models import ModelTrainer
from vis import *
from tqdm import tqdm

# ============================================================================
# CONFIGURATION: Choose what to run
# ============================================================================
RUN_CLASSICAL = False
RUN_BERT = True
RUN_FEATURE_FUSION = False

# XAI CONFIGURATION
RUN_XAI = True
XAI_MODE = 'comprehensive'  # 'quick' or 'comprehensive'

# BERT CONFIGURATION
BERT_CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 128,
    'batch_size': 16,
    'gradient_accumulation_steps': 2,
    'learning_rate': 2e-5,
    'epochs': 7,  # Reduced from 10 due to batch balancing
    'random_state': 42,
    'early_stopping_patience': 2,
    'focal_loss': True,
    'focal_alpha': 0.65,  # Reduced from 0.75 since batch balancing also addresses imbalance
    'focal_gamma': 2.0,
    'freeze_bert_base': True,
    'unfreeze_last_n_layers': 4,
    'dropout_rate': 0.3,
    'prediction_threshold': 0.5,
    'use_batch_balancing': True,  # NEW: Enable batch balancing
}

# ============================================================================
# Models
# ============================================================================
def run_classical_models():
    """Run classical ML models (Random Forest, XGBoost, MLP) with XAI"""
    print("\n" + "=" * 80)
    print("RUNNING CLASSICAL MODELS")
    print("=" * 80 + "\n")

    # 1. Load data
    df = load_data()

    # 2. Feature engineering (includes embeddings)
    text_columns_to_embed = ['notes']
    working_df, unattrib_df = feature_creating(
        df,
        use_embeddings=True,
        text_columns=text_columns_to_embed
    )

    # 3. Train-test split
    X_train, X_test, y_train, y_test = t_t_s(working_df)

    # 4. Undersample training set only
    X_train_und, y_train_und = undersample_train(
        X_train, y_train,
        final_majority_size=2500
    )

    # 5. Train models
    trainer = ModelTrainer()
    trainer.fit_all(
        X_train_und, y_train_und,
        models=['rfc', 'xgb', 'mlpc'],
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )

    # 6. Evaluate on test set
    trainer.evaluate(X_test, y_test)

    # 7. Results summary
    results_df = trainer.get_results()
    print("\n" + "=" * 80)
    print("CLASSICAL MODELS RESULTS")
    print("=" * 80)
    print(results_df[['model', 'cv_score', 'test_f1_macro', 'test_accuracy',
                      'test_precision', 'test_recall']])

    # 8. Best model
    best_model, preprocessors, best_stats = trainer.get_best_model(metric='test_f1_macro')
    print("\nBest Classical Model:")
    print(best_stats)

    # 9. Save results
    results_df.to_csv("classical_results.csv", index=False)
    print("\nResults saved to 'classical_results.csv'")

    # 10. Generate visualizations
    plot_model_performance(results_df, metric='test_f1_macro')
    plot_confusion_matrix_best(trainer, X_test, y_test)
    plot_roc_pr(trainer, X_test, y_test)

    # XAI ANALYSIS
    if RUN_XAI:
        from xai_explanations import (
            explain_all_models,
            explain_samples_with_lime
        )

        # Get feature names
        feature_names = X_train.columns.tolist()

        print("\n" + "=" * 80)
        print("STARTING XAI ANALYSIS")
        print("=" * 80)

        if XAI_MODE == 'quick':
            results = explain_all_models(
                trainer=trainer,
                X_train=X_train_und,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                include_shap=False,
                top_n=15
            )

        elif XAI_MODE == 'comprehensive':
            results = explain_all_models(
                trainer=trainer,
                X_train=X_train_und,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                include_shap=True,
                top_n=20
            )

            explain_samples_with_lime(
                trainer=trainer,
                X_train=X_train_und,
                X_test=X_test,
                feature_names=feature_names,
                sample_indices=[0, 1, 2],
            )

        print("\n" + "=" * 80)
        print("XAI ANALYSIS COMPLETE")
        print("=" * 80 + "\n")

    return results_df, trainer, X_test, y_test


def run_bert_model():
    """Run BERT model"""
    print("\n" + "=" * 80)
    print("RUNNING BERT MODEL")
    print("=" * 80 + "\n")

    from non_classical import BERTClassifier
    from sklearn.model_selection import train_test_split

    # 1. Load data
    print("\nLoading data...")
    df = load_data()

    # 2. Feature engineering (no embeddings needed for BERT)
    print("Creating features...")
    working_df, unattrib_df = feature_creating(
        df,
        use_embeddings=False,
        text_columns=None
    )

    # 3. Get text and apply masking
    X_text = working_df['notes'].fillna('')
    print("\n" + "=" * 60)
    print("APPLYING SEMANTIC MASKING")
    print("=" * 60)
    X_text = X_text.apply(mask_group_names)
    X_text = X_text.apply(mask_location_names)
    print("Semantic masking complete")
    print("=" * 60 + "\n")

    y = working_df['target']

    # 4. Train-test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # 5. Create validation split from training data
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_text, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    print(f"\nData splits:")
    print(f"Train: {len(X_train_split)}")
    print(f"Validation: {len(X_val_split)}")
    print(f"Test: {len(X_test_text)}")

    # 6. Initialize and train BERT model
    bert_model = BERTClassifier(**BERT_CONFIG)
    bert_model.fit(X_train_split, y_train_split, X_val_split, y_val_split)

    # 7. Find optimal threshold
    bert_model.find_optimal_threshold(X_val_split, y_val_split)

    # 8. Evaluate on test set
    results, y_pred, y_proba = bert_model.evaluate(X_test_text, y_test)

    # 9. Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv("bert_results.csv", index=False)
    print("\nResults saved to 'bert_results.csv'")

    # 10. Save training stats
    training_stats = bert_model.get_training_stats()
    training_stats.to_csv("bert_training_stats.csv", index=False)
    print("Training stats saved to 'bert_training_stats.csv'")

    # 11. Visualizations
    from vis import plot_bert_confusion_matrix, plot_bert_roc_pr
    plot_bert_confusion_matrix(y_test, y_pred, model_name='BERT')
    plot_bert_roc_pr(y_test, y_proba, model_name='BERT')

    # XAI FOR BERT
    if RUN_XAI:
        from xai_explanations import explain_bert_model, explain_bert_global

        print("\n" + "=" * 80)
        print("XAI FOR BERT MODEL")
        print("=" * 80)

        # Local explainability
        print("\nLocal Explainability (Sample-Level)")
        explain_bert_model(
            bert_model=bert_model,
            X_test_text=X_test_text,
            y_test=y_test,
            sample_indices=[0, 1, 2, 5, 10],
            model_name='BERT'
        )

        # Global explainability
        print("\nGlobal Explainability (Dataset-Level)")
        explain_bert_global(
            bert_model=bert_model,
            X_test_text=X_test_text,
            y_test=y_test,
            n_samples=100,
            num_features=30,
            model_name='BERT'
        )

        print("\n" + "=" * 80)
        print("XAI COMPLETE")
        print("=" * 80 + "\n")

    return results_df, bert_model, X_test_text, y_test, y_pred, y_proba


def run_feature_fusion():
    """Run BERT + manual features fusion model"""
    print("\n" + "=" * 80)
    print("RUNNING FEATURE FUSION MODEL")
    print("=" * 80 + "\n")

    from feature_fusion import train_fusion_model

    # 1. Load data
    df = load_data()

    # 2. Feature engineering WITH embeddings for manual features
    working_df, unattrib_df = feature_creating(
        df,
        use_embeddings=True,
        text_columns=['notes']
    )

    # 3. Get text and manual features
    X_text = working_df['notes'].fillna('')
    print("\nApplying semantic masking to text...")
    X_text = X_text.apply(mask_group_names)
    X_text = X_text.apply(mask_location_names)
    print("Semantic masking complete")

    # Get manual features (all except text and target)
    manual_feature_cols = [col for col in working_df.columns
                           if col not in ['notes', 'target']]
    X_manual = working_df[manual_feature_cols]
    y = working_df['target']

    # 4. Train-test split
    from sklearn.model_selection import train_test_split
    X_text_train, X_text_test, X_man_train, X_man_test, y_train, y_test = train_test_split(
        X_text, X_manual, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_text_train)}")
    print(f"Test samples: {len(X_text_test)}")
    print(f"Manual features: {X_manual.shape[1]}")

    # 5. Train fusion model
    fusion_model, results = train_fusion_model(
        X_text_train, X_man_train, y_train,
        X_text_test, X_man_test, y_test
    )

    # 6. Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv("feature_fusion_results.csv", index=False)
    print("\nResults saved to 'feature_fusion_results.csv'")

    # XAI FOR FEATURE FUSION
    if RUN_XAI:
        from xai_explanations import explain_feature_fusion_global

        print("\n" + "=" * 80)
        print("XAI FOR FEATURE FUSION")
        print("=" * 80)

        # Get manual feature names
        manual_feature_cols = [col for col in X_manual.columns]

        # Run global XAI
        results = explain_feature_fusion_global(
            fusion_model=fusion_model,
            X_text_test=X_text_test,
            X_features_test=X_man_test,
            y_test=y_test,
            feature_names=manual_feature_cols,
            n_samples=100,
            model_name='FeatureFusion'
        )

        print("\n" + "=" * 80)
        print("XAI COMPLETE")
        print("=" * 80 + "\n")

    return results_df, fusion_model, X_text_test, X_man_test, y_test


def main():
    """Main execution function"""

    print("\n" + "=" * 80)
    print("MACHINE LEARNING PIPELINE")
    print("=" * 80)
    print(f"Classical Models: {RUN_CLASSICAL}")
    print(f"BERT Model: {RUN_BERT}")
    print(f"Feature Fusion: {RUN_FEATURE_FUSION}")
    print(f"XAI Enabled: {RUN_XAI}")
    if RUN_XAI:
        print(f"XAI Mode: {XAI_MODE}")
    print("=" * 80 + "\n")

    # Run classical models if configured
    if RUN_CLASSICAL:
        classical_results, trainer, X_test, y_test = run_classical_models()

    # Run BERT if configured
    if RUN_BERT:
        bert_results, bert_model, X_test_text, y_test_bert, y_pred, y_proba = run_bert_model()

    # Run Feature Fusion if configured
    if RUN_FEATURE_FUSION:
        fusion_results, fusion_model, X_text_test, X_man_test, y_test_fusion = run_feature_fusion()

    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED")
    print("=" * 80)
    print("\nOutputs:")
    print("  - Model results: *.csv files")
    print("  - Visualizations: ./visualizations/")
    if RUN_XAI:
        print("  - XAI Explanations: ./xai_explanations/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()




# import pandas as pd
# import numpy as np
# import torch
# from data_loader import load_data
# from feature_eng import feature_creating
# from feature_eng import mask_group_names, mask_location_names
# from train_test_split import t_t_s
# from under_over import undersample_train
# from models import ModelTrainer
# from vis import *
# from tqdm import tqdm
#
# # ============================================================================
# # CONFIGURATION: Choose what to run
# # ============================================================================
# RUN_CLASSICAL = False
# RUN_BERT = True
# RUN_FEATURE_FUSION = False
#
# # XAI CONFIGURATION
# RUN_XAI = True
# XAI_MODE = 'comprehensive'  # 'quick' or 'comprehensive'
#
# # BERT CONFIGURATION
# BERT_CONFIG = {
#     'model_name': 'bert-base-uncased',
#     'max_length': 128,
#     'batch_size': 16,
#     'gradient_accumulation_steps': 2,
#     'learning_rate': 2e-5,
#     'epochs': 10,
#     'random_state': 42,
#     'early_stopping_patience': 2,
#     'focal_loss': True,
#     'focal_alpha': 0.75, #change from .25
#     'focal_gamma': 2.0,
#     'freeze_bert_base': True,
#     'unfreeze_last_n_layers': 4,
#     'dropout_rate': 0.3,
#     'prediction_threshold': 0.5,
# }
#
# # ============================================================================
# # Models
# # ============================================================================
# def run_classical_models():
#     """Run classical ML models (Random Forest, XGBoost, MLP) with XAI"""
#     print("\n" + "=" * 80)
#     print("RUNNING CLASSICAL MODELS")
#     print("=" * 80 + "\n")
#
#     # 1. Load data
#     df = load_data()
#
#     # 2. Feature engineering (includes embeddings)
#     text_columns_to_embed = ['notes']
#     working_df, unattrib_df = feature_creating(
#         df,
#         use_embeddings=True,
#         text_columns=text_columns_to_embed
#     )
#
#     # 3. Train-test split
#     X_train, X_test, y_train, y_test = t_t_s(working_df)
#
#     # 4. Undersample training set only
#     X_train_und, y_train_und = undersample_train(
#         X_train, y_train,
#         final_majority_size=2500
#     )
#
#     # 5. Train models
#     trainer = ModelTrainer()
#     trainer.fit_all(
#         X_train_und, y_train_und,
#         models=['rfc', 'xgb', 'mlpc'],
#         cv=5,
#         scoring='f1_macro',
#         n_jobs=-1
#     )
#
#     # 6. Evaluate on test set
#     trainer.evaluate(X_test, y_test)
#
#     # 7. Results summary
#     results_df = trainer.get_results()
#     print("\n" + "=" * 80)
#     print("CLASSICAL MODELS RESULTS")
#     print("=" * 80)
#     print(results_df[['model', 'cv_score', 'test_f1_macro', 'test_accuracy',
#                       'test_precision', 'test_recall']])
#
#     # 8. Best model
#     best_model, preprocessors, best_stats = trainer.get_best_model(metric='test_f1_macro')
#     print("\nBest Classical Model:")
#     print(best_stats)
#
#     # 9. Save results
#     results_df.to_csv("classical_results.csv", index=False)
#     print("\nResults saved to 'classical_results.csv'")
#
#     # 10. Generate visualizations
#     plot_model_performance(results_df, metric='test_f1_macro')
#     plot_confusion_matrix_best(trainer, X_test, y_test)
#     plot_roc_pr(trainer, X_test, y_test)
#
#     # XAI ANALYSIS
#     if RUN_XAI:
#         from xai_explanations import (
#             explain_all_models,
#             explain_samples_with_lime
#         )
#
#         # Get feature names
#         feature_names = X_train.columns.tolist()
#
#         print("\n" + "=" * 80)
#         print("STARTING XAI ANALYSIS")
#         print("=" * 80)
#
#         if XAI_MODE == 'quick':
#             results = explain_all_models(
#                 trainer=trainer,
#                 X_train=X_train_und,
#                 X_test=X_test,
#                 y_test=y_test,
#                 feature_names=feature_names,
#                 include_shap=False,
#                 top_n=15
#             )
#
#         elif XAI_MODE == 'comprehensive':
#             results = explain_all_models(
#                 trainer=trainer,
#                 X_train=X_train_und,
#                 X_test=X_test,
#                 y_test=y_test,
#                 feature_names=feature_names,
#                 include_shap=True,
#                 top_n=20
#             )
#
#             explain_samples_with_lime(
#                 trainer=trainer,
#                 X_train=X_train_und,
#                 X_test=X_test,
#                 feature_names=feature_names,
#                 sample_indices=[0, 1, 2],
#             )
#
#         print("\n" + "=" * 80)
#         print("XAI ANALYSIS COMPLETE")
#         print("=" * 80 + "\n")
#
#     return results_df, trainer, X_test, y_test
#
#
# def run_bert_model():
#     """Run BERT model"""
#     print("\n" + "=" * 80)
#     print("RUNNING BERT MODEL")
#     print("=" * 80 + "\n")
#
#     from non_classical import BERTClassifier
#     from sklearn.model_selection import train_test_split
#
#     # 1. Load data
#     print("\nLoading data...")
#     df = load_data()
#
#     # 2. Feature engineering (no embeddings needed for BERT)
#     print("Creating features...")
#     working_df, unattrib_df = feature_creating(
#         df,
#         use_embeddings=False,
#         text_columns=None
#     )
#
#     # 3. Get text and apply masking
#     X_text = working_df['notes'].fillna('')
#     print("\n" + "=" * 60)
#     print("APPLYING SEMANTIC MASKING")
#     print("=" * 60)
#     X_text = X_text.apply(mask_group_names)
#     X_text = X_text.apply(mask_location_names)
#     print("Semantic masking complete")
#     print("=" * 60 + "\n")
#
#     y = working_df['target']
#
#     # 4. Train-test split
#     X_train_text, X_test_text, y_train, y_test = train_test_split(
#         X_text, y,
#         test_size=0.3,
#         random_state=42,
#         stratify=y
#     )
#
#     # 5. Create validation split from training data
#     X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
#         X_train_text, y_train,
#         test_size=0.2,
#         random_state=42,
#         stratify=y_train
#     )
#
#     print(f"\nData splits:")
#     print(f"Train: {len(X_train_split)}")
#     print(f"Validation: {len(X_val_split)}")
#     print(f"Test: {len(X_test_text)}")
#
#     # 6. Initialize and train BERT model
#     bert_model = BERTClassifier(**BERT_CONFIG)
#     bert_model.fit(X_train_split, y_train_split, X_val_split, y_val_split)
#
#     # 7. Find optimal threshold
#     bert_model.find_optimal_threshold(X_val_split, y_val_split)
#
#     # 8. Evaluate on test set
#     results, y_pred, y_proba = bert_model.evaluate(X_test_text, y_test)
#
#     # 9. Save results
#     results_df = pd.DataFrame([results])
#     results_df.to_csv("bert_results.csv", index=False)
#     print("\nResults saved to 'bert_results.csv'")
#
#     # 10. Save training stats
#     training_stats = bert_model.get_training_stats()
#     training_stats.to_csv("bert_training_stats.csv", index=False)
#     print("Training stats saved to 'bert_training_stats.csv'")
#
#     # 11. Visualizations
#     from vis import plot_bert_confusion_matrix, plot_bert_roc_pr
#     plot_bert_confusion_matrix(y_test, y_pred, model_name='BERT')
#     plot_bert_roc_pr(y_test, y_proba, model_name='BERT')
#
#     # XAI FOR BERT
#     if RUN_XAI:
#         from xai_explanations import explain_bert_model, explain_bert_global
#
#         print("\n" + "=" * 80)
#         print("XAI FOR BERT MODEL")
#         print("=" * 80)
#
#         # Local explainability
#         print("\nLocal Explainability (Sample-Level)")
#         explain_bert_model(
#             bert_model=bert_model,
#             X_test_text=X_test_text,
#             y_test=y_test,
#             sample_indices=[0, 1, 2, 5, 10],
#             model_name='BERT'
#         )
#
#         # Global explainability
#         print("\nGlobal Explainability (Dataset-Level)")
#         explain_bert_global(
#             bert_model=bert_model,
#             X_test_text=X_test_text,
#             y_test=y_test,
#             n_samples=100,
#             num_features=30,
#             model_name='BERT'
#         )
#
#         print("\n" + "=" * 80)
#         print("XAI COMPLETE")
#         print("=" * 80 + "\n")
#
#     return results_df, bert_model, X_test_text, y_test, y_pred, y_proba
#
#
# def run_feature_fusion():
#     """Run BERT + manual features fusion model"""
#     print("\n" + "=" * 80)
#     print("RUNNING FEATURE FUSION MODEL")
#     print("=" * 80 + "\n")
#
#     from feature_fusion import train_fusion_model
#
#     # 1. Load data
#     df = load_data()
#
#     # 2. Feature engineering WITH embeddings for manual features
#     working_df, unattrib_df = feature_creating(
#         df,
#         use_embeddings=True,
#         text_columns=['notes']
#     )
#
#     # 3. Get text and manual features
#     X_text = working_df['notes'].fillna('')
#     print("\nApplying semantic masking to text...")
#     X_text = X_text.apply(mask_group_names)
#     X_text = X_text.apply(mask_location_names)
#     print("Semantic masking complete")
#
#     # Get manual features (all except text and target)
#     manual_feature_cols = [col for col in working_df.columns
#                            if col not in ['notes', 'target']]
#     X_manual = working_df[manual_feature_cols]
#     y = working_df['target']
#
#     # 4. Train-test split
#     from sklearn.model_selection import train_test_split
#     X_text_train, X_text_test, X_man_train, X_man_test, y_train, y_test = train_test_split(
#         X_text, X_manual, y,
#         test_size=0.4,
#         random_state=42,
#         stratify=y
#     )
#
#     print(f"\nTraining samples: {len(X_text_train)}")
#     print(f"Test samples: {len(X_text_test)}")
#     print(f"Manual features: {X_manual.shape[1]}")
#
#     # 5. Train fusion model
#     fusion_model, results = train_fusion_model(
#         X_text_train, X_man_train, y_train,
#         X_text_test, X_man_test, y_test
#     )
#
#     # 6. Save results
#     results_df = pd.DataFrame([results])
#     results_df.to_csv("feature_fusion_results.csv", index=False)
#     print("\nResults saved to 'feature_fusion_results.csv'")
#
#     # XAI FOR FEATURE FUSION
#     if RUN_XAI:
#         from xai_explanations import explain_feature_fusion_global
#
#         print("\n" + "=" * 80)
#         print("XAI FOR FEATURE FUSION")
#         print("=" * 80)
#
#         # Get manual feature names
#         manual_feature_cols = [col for col in X_manual.columns]
#
#         # Run global XAI
#         results = explain_feature_fusion_global(
#             fusion_model=fusion_model,
#             X_text_test=X_text_test,
#             X_features_test=X_man_test,
#             y_test=y_test,
#             feature_names=manual_feature_cols,
#             n_samples=100,
#             model_name='FeatureFusion'
#         )
#
#         print("\n" + "=" * 80)
#         print("XAI COMPLETE")
#         print("=" * 80 + "\n")
#
#     return results_df, fusion_model, X_text_test, X_man_test, y_test
#
#
# def main():
#     """Main execution function"""
#
#     print("\n" + "=" * 80)
#     print("MACHINE LEARNING PIPELINE")
#     print("=" * 80)
#     print(f"Classical Models: {RUN_CLASSICAL}")
#     print(f"BERT Model: {RUN_BERT}")
#     print(f"Feature Fusion: {RUN_FEATURE_FUSION}")
#     print(f"XAI Enabled: {RUN_XAI}")
#     if RUN_XAI:
#         print(f"XAI Mode: {XAI_MODE}")
#     print("=" * 80 + "\n")
#
#     # Run classical models if configured
#     if RUN_CLASSICAL:
#         classical_results, trainer, X_test, y_test = run_classical_models()
#
#     # Run BERT if configured
#     if RUN_BERT:
#         bert_results, bert_model, X_test_text, y_test_bert, y_pred, y_proba = run_bert_model()
#
#     # Run Feature Fusion if configured
#     if RUN_FEATURE_FUSION:
#         fusion_results, fusion_model, X_text_test, X_man_test, y_test_fusion = run_feature_fusion()
#
#     print("\n" + "=" * 80)
#     print("ALL TASKS COMPLETED")
#     print("=" * 80)
#     print("\nOutputs:")
#     print("  - Model results: *.csv files")
#     print("  - Visualizations: ./visualizations/")
#     if RUN_XAI:
#         print("  - XAI Explanations: ./xai_explanations/")
#     print("=" * 80 + "\n")
#
#
# if __name__ == "__main__":
#     main()