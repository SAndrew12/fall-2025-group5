import pandas as pd
import torch  # For LR finder DataLoader
from data_loader import load_data
from feature_eng import feature_creating
from feature_eng import mask_group_names, mask_location_names
from train_test_split import t_t_s
from under_over import undersample_train
from models import ModelTrainer
from vis import *

# ============================================================================
# CONFIGURATION: Choose what to run
# ============================================================================
RUN_CLASSICAL = False
RUN_BERT = True
RUN_FEATURE_FUSION = False

# XAI CONFIGURATION
RUN_XAI = True  # Toggle XAI on/off
XAI_MODE = 'comprehensive'  # 'quick' or 'comprehensive'

# BERT CONFIGURATION
BERT_CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 128,  # Will be validated against actual text lengths
    'batch_size': 16,  # Physical batch size
    'gradient_accumulation_steps': 2,  # Effective batch size = 16 * 2 = 32
    'learning_rate': 2e-5,  # Will be validated with LR finder if enabled
    'epochs': 5,
    'random_state': 42,

    # Class imbalance strategy: Choose ONE of: 'balanced_sampler', 'class_weights', or 'none'
    'class_imbalance_strategy': 'balanced_sampler',  # NEW: unified parameter

    # Advanced settings
    'focal_loss': False,
    'focal_alpha': 0.75,  # If using focal loss
    'focal_gamma': 2.0,
    'freeze_bert_base': True,
    'unfreeze_last_n_layers': 2,  # REDUCED from 4
    'label_smoothing': 0.1,
    'dropout_rate': 0.5,  # INCREASED regularization
    'beta_class_weights': 0.99,  # More moderate than 0.9999
    'prediction_threshold': 0.5,

    # Diagnostic tools
    'run_lr_finder': True,  # Set to True to find optimal LR before training (will prompt for user input)
    'analyze_text_lengths': True,  # Analyze and suggest optimal max_length
}


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

    # 5. Train models (with in-CV oversampling)
    trainer = ModelTrainer()
    trainer.fit_all(
        X_train_und, y_train_und,
        models=['rfc', 'xgb', 'mlpc'],
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )

    # 6. Evaluate on untouched test set
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

    # ========================================================================
    # XAI ANALYSIS
    # ========================================================================
    if RUN_XAI:
        from xai_explanations import (
            explain_all_models,
            explain_samples_with_lime
        )

        # Get feature names
        feature_names = X_train.columns.tolist()

        print("\n" + "=" * 80)
        print("STARTING ENHANCED XAI ANALYSIS")
        print("=" * 80)

        if XAI_MODE == 'quick':
            # Quick mode: Feature importance only (no SHAP)
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
            # Comprehensive mode: Everything including SHAP
            results = explain_all_models(
                trainer=trainer,
                X_train=X_train_und,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                include_shap=True,
                top_n=20
            )

            # Optional: LIME for specific samples
            print("\n(Optional) Explaining specific samples with LIME...")
            explain_samples_with_lime(
                trainer=trainer,
                X_train=X_train_und,
                X_test=X_test,
                feature_names=feature_names,
                sample_indices=[0, 1, 2],
            )

        print("\n" + "=" * 80)
        print("XAI ANALYSIS COMPLETE!")
        print("=" * 80 + "\n")

    return results_df, trainer, X_test, y_test


def run_bert_model():
    """Run BERT model with improved configuration and diagnostic tools"""
    print("\n" + "=" * 80)
    print("RUNNING BERT MODEL WITH IMPROVED CONFIGURATION")
    print("=" * 80 + "\n")

    # Import improved BERT classifier and utilities
    from non_classical import BERTClassifier, find_learning_rate, plot_lr_finder
    from sklearn.model_selection import train_test_split

    # 1. Load data
    df = load_data()

    # 2. Feature engineering (no embeddings needed for BERT)
    working_df, unattrib_df = feature_creating(
        df,
        use_embeddings=False,
        text_columns=None
    )

    # 3. Get the text column and labels with semantic masking
    X_text = working_df['notes'].fillna('')
    print("\n" + "=" * 60)
    print("APPLYING SEMANTIC MASKING")
    print("=" * 60)
    print("Replacing all group names with [ARMED_GROUP]")
    print("Replacing locations with [LOCATION]")
    X_text = X_text.apply(mask_group_names)
    X_text = X_text.apply(mask_location_names)
    print("Semantic masking complete!")
    print("=" * 60 + "\n")

    # Verify masking worked
    print("=== CHECKING FOR LEAKAGE ===")
    sample_texts = X_text.head(10)
    for i, text in enumerate(sample_texts):
        has_taliban = any(word in text.lower() for word in ['taliban', 'taleban'])
        has_isis = any(word in text.lower() for word in ['isis', 'islamic state'])
        has_placeholder = '[ARMED_GROUP' in text or '[LOCATION]' in text
        print(f"Text {i}: Taliban={has_taliban}, ISIS={has_isis}, HasPlaceholder={has_placeholder}")
        if has_taliban or has_isis:
            print(f"  ⚠️ LEAKAGE: {text[:100]}...")
        elif has_placeholder:
            print(f"  ✓ Masked: {text[:100]}...")
    print("=== END CHECK ===\n")

    y = working_df['target']

    # 4. Train-test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    print(f"\nDataset sizes:")
    print(f"  Training samples: {len(X_train_text)}")
    print(f"  Test samples: {len(X_test_text)}")
    print(f"  Class distribution (train): {y_train.value_counts().to_dict()}")
    print(f"  Class distribution (test): {y_test.value_counts().to_dict()}")

    # ========================================================================
    # INITIALIZE BERT MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("INITIALIZING BERT MODEL")
    print("=" * 80)
    print(f"Strategy: {BERT_CONFIG['class_imbalance_strategy']}")
    print(f"Batch size: {BERT_CONFIG['batch_size']}")
    print(f"Gradient accumulation: {BERT_CONFIG['gradient_accumulation_steps']}")
    print(f"Effective batch size: {BERT_CONFIG['batch_size'] * BERT_CONFIG['gradient_accumulation_steps']}")
    print(f"Unfrozen layers: {BERT_CONFIG['unfreeze_last_n_layers']}")
    print(f"Dropout rate: {BERT_CONFIG['dropout_rate']}")
    print("=" * 80)

    bert_model = BERTClassifier(
        model_name=BERT_CONFIG['model_name'],
        max_length=BERT_CONFIG['max_length'],
        batch_size=BERT_CONFIG['batch_size'],
        learning_rate=BERT_CONFIG['learning_rate'],
        epochs=BERT_CONFIG['epochs'],
        random_state=BERT_CONFIG['random_state'],
        class_imbalance_strategy=BERT_CONFIG['class_imbalance_strategy'],
        focal_loss=BERT_CONFIG['focal_loss'],
        focal_alpha=BERT_CONFIG['focal_alpha'],
        focal_gamma=BERT_CONFIG['focal_gamma'],
        freeze_bert_base=BERT_CONFIG['freeze_bert_base'],
        unfreeze_last_n_layers=BERT_CONFIG['unfreeze_last_n_layers'],
        label_smoothing=BERT_CONFIG['label_smoothing'],
        dropout_rate=BERT_CONFIG['dropout_rate'],
        gradient_accumulation_steps=BERT_CONFIG['gradient_accumulation_steps'],
        beta_class_weights=BERT_CONFIG['beta_class_weights'],
        prediction_threshold=BERT_CONFIG['prediction_threshold']
    )

    # ========================================================================
    # DIAGNOSTIC: TEXT LENGTH ANALYSIS
    # ========================================================================
    if BERT_CONFIG['analyze_text_lengths']:
        stats = bert_model.analyze_text_lengths(X_train_text)

        # Auto-adjust max_length if much too long
        suggested = int(stats['95th_percentile']) + 10
        if BERT_CONFIG['max_length'] > suggested * 1.5:
            print(f"\n⚠️  Adjusting max_length from {BERT_CONFIG['max_length']} to {suggested}")
            bert_model.max_length = suggested

    # Split train into train/val
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_text, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    # ========================================================================
    # DIAGNOSTIC: LEARNING RATE FINDER
    # ========================================================================
    if BERT_CONFIG['run_lr_finder']:
        print("\nRunning learning rate finder...")

        # Create temporary model and dataloader for LR finding
        temp_model = bert_model._create_model()
        from non_classical import TextDataset
        temp_dataset = TextDataset(
            X_train_split.tolist()[:500],  # Use subset for speed
            y_train_split.tolist()[:500],
            bert_model.tokenizer,
            bert_model.max_length
        )
        temp_loader = torch.utils.data.DataLoader(
            temp_dataset,
            batch_size=bert_model.batch_size,
            shuffle=True
        )

        lrs, losses, suggested_lr = find_learning_rate(
            temp_model, temp_loader,
            start_lr=1e-7, end_lr=1e-2, num_iter=100
        )

        plot_lr_finder(lrs, losses, suggested_lr, 'lr_finder_bert.png')

        # Ask user if they want to use suggested LR
        print(f"\nCurrent LR: {bert_model.learning_rate:.2e}")
        print(f"Suggested LR: {suggested_lr:.2e}")
        response = input("Use suggested LR? (y/n): ")
        if response.lower() == 'y':
            bert_model.learning_rate = suggested_lr
            print(f"✓ Updated learning rate to {suggested_lr:.2e}")

    # ========================================================================
    # TRAINING
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING BERT MODEL")
    print("=" * 80)

    bert_model.fit(X_train_split, y_train_split, X_val_split, y_val_split)
    bert_model.find_optimal_threshold(X_val_split, y_val_split)
    results, y_pred, y_proba = bert_model.evaluate(X_test_text, y_test)

    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv("bert_results.csv", index=False)
    print("\nResults saved to 'bert_results.csv'")

    # Save training stats
    training_stats = bert_model.get_training_stats()
    training_stats.to_csv("bert_training_stats.csv", index=False)
    print("Training stats saved to 'bert_training_stats.csv'")

    # Visualizations
    from vis import plot_bert_confusion_matrix, plot_bert_roc_pr
    plot_bert_confusion_matrix(y_test, y_pred, model_name='BERT')
    plot_bert_roc_pr(y_test, y_proba, model_name='BERT')

    # ========================================================================
    # XAI FOR BERT
    # ========================================================================
    if RUN_XAI:
        from xai_explanations import explain_bert_model, explain_bert_global

        # Local explainability
        print("\n--- LOCAL Explainability (Sample-Level) ---")
        explain_bert_model(
            bert_model=bert_model,
            X_test_text=X_test_text,
            y_test=y_test,
            sample_indices=[0, 1, 2, 5, 10],
            model_name='BERT'
        )

        # Global explainability
        print("\n--- GLOBAL Explainability (Dataset-Level) ---")
        explain_bert_global(
            bert_model=bert_model,
            X_test_text=X_test_text,
            y_test=y_test,
            n_samples=100,
            num_features=30,
            model_name='BERT'
        )

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
    print("Semantic masking complete!")

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

    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv("feature_fusion_results.csv", index=False)
    print("\nResults saved to 'feature_fusion_results.csv'")

    # ========================================================================
    # XAI FOR FEATURE FUSION
    # ========================================================================
    if RUN_XAI:
        from xai_explanations import explain_feature_fusion_global

        print("\n" + "=" * 80)
        print("GLOBAL XAI FOR FEATURE FUSION")
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

        print("\nFeature Fusion XAI Complete!")
        print("=" * 80 + "\n")

    return results_df, fusion_model, X_text_test, X_man_test, y_test


def main():
    """Main execution function"""

    print("\n" + "=" * 80)
    print("MACHINE LEARNING PIPELINE WITH IMPROVED BERT")
    print("=" * 80)
    print(f"Classical Models: {RUN_CLASSICAL}")
    print(f"BERT Model: {RUN_BERT}")
    print(f"Feature Fusion: {RUN_FEATURE_FUSION}")
    print(f"XAI Enabled: {RUN_XAI}")
    if RUN_XAI:
        print(f"XAI Mode: {XAI_MODE}")
    if RUN_BERT:
        print(f"\nBERT Configuration:")
        print(f"  Strategy: {BERT_CONFIG['class_imbalance_strategy']}")
        print(f"  Effective batch size: {BERT_CONFIG['batch_size'] * BERT_CONFIG['gradient_accumulation_steps']}")
        print(f"  Unfrozen layers: {BERT_CONFIG['unfreeze_last_n_layers']}")
        print(f"  Dropout: {BERT_CONFIG['dropout_rate']}")
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
    if RUN_BERT and BERT_CONFIG['run_lr_finder']:
        print("  - LR Finder plot: lr_finder_bert.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()