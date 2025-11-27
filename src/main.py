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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from save_trained_models import (
    save_classical_models,
    save_bert_model,
    save_feature_fusion_model
)

# ============================================================================
# CONFIGURATION: Choose what to run
# ============================================================================
RUN_CLASSICAL = True
RUN_BERT = False
RUN_FEATURE_FUSION = False

# XAI CONFIGURATION
RUN_XAI = True
XAI_MODE = 'comprehensive'  # 'quick' or 'comprehensive'

# BERT CONFIGURATION (Applied to both BERT and Feature Fusion)
BERT_CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 320,
    'batch_size': 16,
    'gradient_accumulation_steps': 2,
    'learning_rate': 1e-5,
    'epochs': 12,
    'random_state': 42,
    'early_stopping_patience': 3,
    'focal_loss': True,
    'focal_alpha': 0.7,
    'focal_gamma': 2.5,
    'freeze_bert_base': False,
    'unfreeze_last_n_layers': 12,
    'dropout_rate': 0.3,
    'prediction_threshold': 0.5,
    'use_batch_balancing': True,
}

# FEATURE FUSION SPECIFIC CONFIG
FUSION_CONFIG = {
    **BERT_CONFIG,  # Inherit all BERT config
    'hidden_dim': 128,  # Hidden dimension for manual features
}

# Manual features to use in feature fusion
FUSION_MANUAL_FEATURES = [
    # Base numeric features
    'civilian_targeting',
    'fatalities',
    'violence_against_women',

    # Lean features (casualty thresholds)
    # 'has_casualties',
    # 'high_casualties',
    # 'very_high_casualties',
    # 'zero_fatalities',

    # Lean features (attack patterns)
    # 'coordinated_attack',
    # 'series_attack',

    # One-hot encoded sub_event_type columns
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
    'sub_event_type_Suicide bomb'
]


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
    """Run BERT + manual features fusion model with improved BERT configuration"""
    print("\n" + "=" * 80)
    print("RUNNING FEATURE FUSION MODEL (Improved BERT Config)")
    print("=" * 80 + "\n")

    from feature_fusion import BERTFeatureFusionClassifier
    from sklearn.model_selection import train_test_split

    # 1. Load data
    print("\nLoading data...")
    df = load_data()

    # 2. Feature engineering (no embeddings needed)
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

    # 4. Select manual features
    available_features = [col for col in FUSION_MANUAL_FEATURES if col in working_df.columns]
    missing_features = [col for col in FUSION_MANUAL_FEATURES if col not in working_df.columns]

    print(f"Features requested: {len(FUSION_MANUAL_FEATURES)}")
    print(f"Features available: {len(available_features)}")

    if missing_features:
        print(f"\nWARNING: {len(missing_features)} features not found:")
        for feat in missing_features[:10]:
            print(f"  - {feat}")
        if len(missing_features) > 10:
            print(f"  ... and {len(missing_features) - 10} more")

    X_manual = working_df[available_features].copy()
    print(f"\nSelected {len(available_features)} manual features")
    print("=" * 60 + "\n")

    y = working_df['target']

    # 5. Train-test split
    X_text_train, X_text_test, X_man_train, X_man_test, y_train, y_test = train_test_split(
        X_text, X_manual, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # IMPORTANT: scale manual features using training data only
    # This scaler MUST be saved for the demo!
    scaler = StandardScaler()
    X_man_train = pd.DataFrame(
        scaler.fit_transform(X_man_train),
        columns=X_man_train.columns,
        index=X_man_train.index,
    )
    X_man_test = pd.DataFrame(
        scaler.transform(X_man_test),
        columns=X_man_test.columns,
        index=X_man_test.index,
    )

    # 6. Create validation split from training data
    X_text_tr, X_text_val, X_man_tr, X_man_val, y_tr, y_val = train_test_split(
        X_text_train, X_man_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    print(f"Data splits:")
    print(f"Train: {len(X_text_tr)}")
    print(f"Validation: {len(X_text_val)}")
    print(f"Test: {len(X_text_test)}")
    print(f"Manual features: {X_manual.shape[1]}")

    # 7. Initialize and train feature fusion model with improved config
    print("\n" + "=" * 80)
    print("USING IMPROVED BERT CONFIGURATION")
    print("=" * 80)
    print(f"Epochs: {FUSION_CONFIG['epochs']}")
    print(f"Early stopping patience: {FUSION_CONFIG['early_stopping_patience']}")
    print(f"Focal loss: {FUSION_CONFIG['focal_loss']}")
    print(f"Batch balancing: {FUSION_CONFIG['use_batch_balancing']}")
    print(f"Freeze BERT base: {FUSION_CONFIG['freeze_bert_base']}")
    print(f"Unfreeze last N layers: {FUSION_CONFIG['unfreeze_last_n_layers']}")
    print(f"Gradient accumulation steps: {FUSION_CONFIG['gradient_accumulation_steps']}")
    print("=" * 80 + "\n")

    fusion_model = BERTFeatureFusionClassifier(**FUSION_CONFIG)
    fusion_model.fit(
        X_text_tr, X_man_tr, y_tr,
        X_text_val, X_man_val, y_val
    )

    # 8. Find optimal threshold
    fusion_model.find_optimal_threshold(X_text_val, X_man_val, y_val)

    # 9. Evaluate on test set
    results, y_pred, y_proba = fusion_model.evaluate(X_text_test, X_man_test, y_test)

    # 10. Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv("feature_fusion_results.csv", index=False)
    print("\nResults saved to 'feature_fusion_results.csv'")

    # 11. Save training stats
    training_stats = fusion_model.get_training_stats()
    training_stats.to_csv("feature_fusion_training_stats.csv", index=False)
    print("Training stats saved to 'feature_fusion_training_stats.csv'")

    # 12. Visualizations
    from vis import plot_bert_confusion_matrix, plot_bert_roc_pr
    plot_bert_confusion_matrix(y_test, y_pred, model_name='Feature Fusion')
    plot_bert_roc_pr(y_test, y_proba, model_name='Feature Fusion')

    # XAI FOR FEATURE FUSION
    if RUN_XAI:
        from xai_explanations import explain_feature_fusion_global

        print("\n" + "=" * 80)
        print("XAI FOR FEATURE FUSION")
        print("=" * 80)

        # Get manual feature names
        manual_feature_cols = list(X_manual.columns)

        # Run global XAI
        xai_results = explain_feature_fusion_global(
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

    # *** CRITICAL FIX: Return the scaler so it can be saved! ***
    return results_df, fusion_model, X_text_test, X_man_test, y_test, y_pred, y_proba, scaler


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

        # *** SAVE CLASSICAL MODELS ***
        print("\n" + "=" * 80)
        print("SAVING CLASSICAL MODELS")
        print("=" * 80)
        feature_names = X_test.columns.tolist()
        save_classical_models(trainer, feature_names, model_dir='saved_models/classical')
        print("=" * 80 + "\n")

    # Run BERT if configured
    if RUN_BERT:
        bert_results, bert_model, X_test_text, y_test_bert, y_pred, y_proba = run_bert_model()

        # *** SAVE BERT MODEL ***
        print("\n" + "=" * 80)
        print("SAVING BERT MODEL")
        print("=" * 80)
        save_bert_model(bert_model, model_dir='saved_models/bert')
        print("=" * 80 + "\n")

    # Run Feature Fusion if configured
    if RUN_FEATURE_FUSION:
        # *** NOTE: Now returns scaler as well! ***
        fusion_results, fusion_model, X_text_test, X_man_test, y_test_fusion, y_pred, y_proba, scaler = run_feature_fusion()

        # *** SAVE FEATURE FUSION MODEL ***
        print("\n" + "=" * 80)
        print("SAVING FEATURE FUSION MODEL")
        print("=" * 80)
        manual_feature_names = X_man_test.columns.tolist()
        save_feature_fusion_model(
            fusion_model,
            manual_feature_names,
            scaler,  # The scaler is critical for demo!
            model_dir='saved_models/feature_fusion'
        )
        print("=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED")
    print("=" * 80)
    print("\nOutputs:")
    print("  - Model results: *.csv files")
    print("  - Visualizations: ./visualizations/")
    print("  - Saved models: ./saved_models/")
    if RUN_XAI:
        print("  - XAI Explanations: ./xai_explanations/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

