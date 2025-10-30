import pandas as pd
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
RUN_BERT = False
RUN_FEATURE_FUSION = True  

# ============================================================================


def run_classical_models():
    """Run classical ML models (Random Forest, XGBoost, MLP)"""
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

    return results_df, trainer, X_test, y_test


def run_bert_model():
    """Run BERT model on text data with improved minority class handling"""
    print("\n" + "=" * 80)
    print("RUNNING BERT MODEL")
    print("=" * 80 + "\n")

    # Import improved BERT classifier
    from non_classical import BERTClassifier

    # 1. Load data
    df = load_data()

    # 2. Feature engineering (no embeddings needed for BERT)
    working_df, unattrib_df = feature_creating(
        df,
        use_embeddings=False,  # BERT handles its own text encoding
        text_columns=None
    )

    # 3. Get the text column and labels
    X_text = working_df['notes'].fillna('')
    print("\nRemoving group names and locations from text...")
    X_text = X_text.apply(mask_group_names)
    X_text = X_text.apply(mask_location_names)
    print("Text masking complete!")

    print("\n=== CHECKING FOR LEAKAGE ===")
    sample_texts = X_text.head(10)
    for i, text in enumerate(sample_texts):
        has_taliban = any(word in text.lower() for word in ['taliban', 'taleban'])
        has_isis = any(word in text.lower() for word in ['isis', 'islamic state'])
        print(f"Text {i}: Taliban={has_taliban}, ISIS={has_isis}")
        if has_taliban or has_isis:
            print(f" LEAKAGE DETECTED: {text[:100]}...")
    print("=== END CHECK ===\n")

    y = working_df['target']

    # 4. Train-test split for text data
    from sklearn.model_selection import train_test_split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train_text)}")
    print(f"Test samples: {len(X_test_text)}")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    print(f"Class distribution (test): {y_test.value_counts().to_dict()}")

    # ============================================================================
    # EXPERIMENT 1: Balanced Sampler + Moderate Class Weights
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Balanced Sampler + Moderate Class Weights")
    print("=" * 80)

    bert_model_1 = BERTClassifier(
        model_name='bert-base-uncased',
        max_length=256,
        batch_size=4,
        learning_rate=2e-5,
        epochs=5,
        random_state=42,
        use_class_weights=True,  # Use moderate effective number weights
        use_balanced_sampler=True,  # Balance batches
        focal_loss=False
    )

    # Split train into train/val with larger validation set
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_text, y_train,
        test_size=0.2,  # Larger validation set
        random_state=42,
        stratify=y_train
    )

    bert_model_1.fit(X_train_split, y_train_split, X_val_split, y_val_split)

    # Find optimal threshold on validation set
    bert_model_1.find_optimal_threshold(X_val_split, y_val_split)

    results_1, y_pred_1, y_proba_1 = bert_model_1.evaluate(X_test_text, y_test)

    # ============================================================================
    # EXPERIMENT 2: Focal Loss (no class weights)
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Focal Loss")
    print("=" * 80)

    bert_model_2 = BERTClassifier(
        model_name='bert-base-uncased',
        max_length=256,
        batch_size=4,
        learning_rate=2e-5,
        epochs=5,
        random_state=42,
        use_class_weights=False,
        use_balanced_sampler=True,
        focal_loss=True,  # Use focal loss
        focal_alpha=0.25,
        focal_gamma=2.0
    )

    bert_model_2.fit(X_train_split, y_train_split, X_val_split, y_val_split)
    bert_model_2.find_optimal_threshold(X_val_split, y_val_split)
    results_2, y_pred_2, y_proba_2 = bert_model_2.evaluate(X_test_text, y_test)

    # ============================================================================
    # EXPERIMENT 3: Balanced Sampler Only (baseline)
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Balanced Sampler Only (baseline)")
    print("=" * 80)

    bert_model_3 = BERTClassifier(
        model_name='bert-base-uncased',
        max_length=256,
        batch_size=4,
        learning_rate=2e-5,
        epochs=5,
        random_state=42,
        use_class_weights=False,
        use_balanced_sampler=True,
        focal_loss=False
    )

    bert_model_3.fit(X_train_split, y_train_split, X_val_split, y_val_split)
    bert_model_3.find_optimal_threshold(X_val_split, y_val_split)
    results_3, y_pred_3, y_proba_3 = bert_model_3.evaluate(X_test_text, y_test)

    # ============================================================================
    # COMPARE RESULTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("COMPARING ALL EXPERIMENTS")
    print("=" * 80)

    comparison_df = pd.DataFrame([
        {
            'experiment': 'Balanced Sampler + Class Weights',
            'macro_f1': results_1['test_f1_macro'],
            'accuracy': results_1['test_accuracy'],
            'minority_recall': results_1['minority_recall'],
            'minority_precision': results_1['minority_precision'],
            'minority_f1': results_1['minority_f1']
        },
        {
            'experiment': 'Focal Loss',
            'macro_f1': results_2['test_f1_macro'],
            'accuracy': results_2['test_accuracy'],
            'minority_recall': results_2['minority_recall'],
            'minority_precision': results_2['minority_precision'],
            'minority_f1': results_2['minority_f1']
        },
        {
            'experiment': 'Balanced Sampler Only',
            'macro_f1': results_3['test_f1_macro'],
            'accuracy': results_3['test_accuracy'],
            'minority_recall': results_3['minority_recall'],
            'minority_precision': results_3['minority_precision'],
            'minority_f1': results_3['minority_f1']
        }
    ])

    print(comparison_df.to_string(index=False))

    # Find best approach
    best_idx = comparison_df['minority_f1'].idxmax()
    best_experiment = comparison_df.iloc[best_idx]

    print("\n" + "=" * 80)
    print(f"BEST APPROACH: {best_experiment['experiment']}")
    print("=" * 80)
    print(f"Minority F1: {best_experiment['minority_f1']:.4f}")
    print(f"Minority Recall: {best_experiment['minority_recall']:.4f}")
    print(f"Minority Precision: {best_experiment['minority_precision']:.4f}")
    print(f"Macro F1: {best_experiment['macro_f1']:.4f}")

    # Save results
    comparison_df.to_csv("bert_experiments_comparison.csv", index=False)
    print("\nComparison saved to 'bert_experiments_comparison.csv'")

    # Save best model's detailed results
    if best_idx == 0:
        best_model = bert_model_1
        best_results = results_1
        best_y_pred = y_pred_1
        best_y_proba = y_proba_1
    elif best_idx == 1:
        best_model = bert_model_2
        best_results = results_2
        best_y_pred = y_pred_2
        best_y_proba = y_proba_2
    else:
        best_model = bert_model_3
        best_results = results_3
        best_y_pred = y_pred_3
        best_y_proba = y_proba_3

    results_df = pd.DataFrame([best_results])
    results_df.to_csv("bert_best_results.csv", index=False)

    training_stats = best_model.get_training_stats()
    training_stats.to_csv("bert_best_training_stats.csv", index=False)

    # Generate visualizations
    from vis import plot_bert_confusion_matrix, plot_bert_roc_pr
    plot_bert_confusion_matrix(y_test, best_y_pred, model_name=f'BERT - {best_experiment["experiment"]}')
    plot_bert_roc_pr(y_test, best_y_proba, model_name=f'BERT - {best_experiment["experiment"]}')

    # Optional: Save best model
    # best_model.save_model("bert_model_best")

    return comparison_df, best_model, X_test_text, y_test, best_y_pred, best_y_proba


def run_feature_fusion_model():
    """Run BERT with manual features fusion"""
    print("\n" + "=" * 80)
    print("RUNNING BERT FEATURE FUSION MODEL")
    print("=" * 80 + "\n")

    # Import feature fusion classifier
    from bert_feature_fusion import BERTFeatureFusionClassifier

    # 1. Load data
    df = load_data()

    # 2. Feature engineering - CREATE BOTH MANUAL FEATURES AND TEXT
    # We need manual features (not embeddings) + text
    working_df, unattrib_df = feature_creating(
        df,
        use_embeddings=False,  # Don't need sentence embeddings
        text_columns=None
    )

    # 3. Get the text column
    X_text = working_df['notes'].fillna('')
    print("\nRemoving group names and locations from text...")
    X_text = X_text.apply(mask_group_names)
    X_text = X_text.apply(mask_location_names)
    print("Text masking complete!")

    # 4. Get manual features (you can customize this list based on your needs)
    # These are the features from feature_eng that aren't embeddings
    manual_feature_cols = [
        'civilian_targeting', 'fatalities', 'violence_against_women',
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

    # Check which features actually exist in the dataframe
    available_manual_features = [col for col in manual_feature_cols if col in working_df.columns]
    print(f"\nUsing {len(available_manual_features)} manual features:")
    print(available_manual_features)

    X_features = working_df[available_manual_features]
    y = working_df['target']

    # 5. Train-test split
    from sklearn.model_selection import train_test_split
    X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
        X_text, X_features, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_text_train)}")
    print(f"Test samples: {len(X_text_test)}")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    print(f"Class distribution (test): {y_test.value_counts().to_dict()}")

    # ============================================================================
    # EXPERIMENT 1: Balanced Sampler + Moderate Class Weights
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Feature Fusion - Balanced Sampler + Class Weights")
    print("=" * 80)

    fusion_model_1 = BERTFeatureFusionClassifier(
        model_name='bert-base-uncased',
        max_length=256,
        batch_size=4,
        learning_rate=2e-5,
        epochs=5,
        random_state=42,
        hidden_dim=128,
        dropout=0.3,
        use_class_weights=True,
        use_balanced_sampler=True,
        focal_loss=False
    )

    # Split train into train/val
    X_text_train_split, X_text_val_split, X_feat_train_split, X_feat_val_split, y_train_split, y_val_split = train_test_split(
        X_text_train, X_feat_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    fusion_model_1.fit(
        X_text_train_split, X_feat_train_split, y_train_split,
        X_text_val_split, X_feat_val_split, y_val_split
    )

    # Find optimal threshold
    fusion_model_1.find_optimal_threshold(X_text_val_split, X_feat_val_split, y_val_split)

    results_1, y_pred_1, y_proba_1 = fusion_model_1.evaluate(X_text_test, X_feat_test, y_test)

    # ============================================================================
    # EXPERIMENT 2: Focal Loss
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Feature Fusion - Focal Loss")
    print("=" * 80)

    fusion_model_2 = BERTFeatureFusionClassifier(
        model_name='bert-base-uncased',
        max_length=256,
        batch_size=4,
        learning_rate=2e-5,
        epochs=5,
        random_state=42,
        hidden_dim=128,
        dropout=0.3,
        use_class_weights=False,
        use_balanced_sampler=True,
        focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0
    )

    fusion_model_2.fit(
        X_text_train_split, X_feat_train_split, y_train_split,
        X_text_val_split, X_feat_val_split, y_val_split
    )
    fusion_model_2.find_optimal_threshold(X_text_val_split, X_feat_val_split, y_val_split)
    results_2, y_pred_2, y_proba_2 = fusion_model_2.evaluate(X_text_test, X_feat_test, y_test)

    # ============================================================================
    # EXPERIMENT 3: Balanced Sampler Only (baseline)
    # ============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Feature Fusion - Balanced Sampler Only")
    print("=" * 80)

    fusion_model_3 = BERTFeatureFusionClassifier(
        model_name='bert-base-uncased',
        max_length=256,
        batch_size=4,
        learning_rate=2e-5,
        epochs=5,
        random_state=42,
        hidden_dim=128,
        dropout=0.3,
        use_class_weights=False,
        use_balanced_sampler=True,
        focal_loss=False
    )

    fusion_model_3.fit(
        X_text_train_split, X_feat_train_split, y_train_split,
        X_text_val_split, X_feat_val_split, y_val_split
    )
    fusion_model_3.find_optimal_threshold(X_text_val_split, X_feat_val_split, y_val_split)
    results_3, y_pred_3, y_proba_3 = fusion_model_3.evaluate(X_text_test, X_feat_test, y_test)

    # ============================================================================
    # COMPARE RESULTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("COMPARING ALL FEATURE FUSION EXPERIMENTS")
    print("=" * 80)

    comparison_df = pd.DataFrame([
        {
            'experiment': 'Feature Fusion - Balanced + Weights',
            'macro_f1': results_1['test_f1_macro'],
            'accuracy': results_1['test_accuracy'],
            'minority_recall': results_1['minority_recall'],
            'minority_precision': results_1['minority_precision'],
            'minority_f1': results_1['minority_f1']
        },
        {
            'experiment': 'Feature Fusion - Focal Loss',
            'macro_f1': results_2['test_f1_macro'],
            'accuracy': results_2['test_accuracy'],
            'minority_recall': results_2['minority_recall'],
            'minority_precision': results_2['minority_precision'],
            'minority_f1': results_2['minority_f1']
        },
        {
            'experiment': 'Feature Fusion - Balanced Only',
            'macro_f1': results_3['test_f1_macro'],
            'accuracy': results_3['test_accuracy'],
            'minority_recall': results_3['minority_recall'],
            'minority_precision': results_3['minority_precision'],
            'minority_f1': results_3['minority_f1']
        }
    ])

    print(comparison_df.to_string(index=False))

    # Find best approach
    best_idx = comparison_df['minority_f1'].idxmax()
    best_experiment = comparison_df.iloc[best_idx]

    print("\n" + "=" * 80)
    print(f"BEST FEATURE FUSION APPROACH: {best_experiment['experiment']}")
    print("=" * 80)
    print(f"Minority F1: {best_experiment['minority_f1']:.4f}")
    print(f"Minority Recall: {best_experiment['minority_recall']:.4f}")
    print(f"Minority Precision: {best_experiment['minority_precision']:.4f}")
    print(f"Macro F1: {best_experiment['macro_f1']:.4f}")

    # Save results
    comparison_df.to_csv("feature_fusion_comparison.csv", index=False)
    print("\nComparison saved to 'feature_fusion_comparison.csv'")

    # Save best model's detailed results
    if best_idx == 0:
        best_model = fusion_model_1
        best_results = results_1
        best_y_pred = y_pred_1
        best_y_proba = y_proba_1
    elif best_idx == 1:
        best_model = fusion_model_2
        best_results = results_2
        best_y_pred = y_pred_2
        best_y_proba = y_proba_2
    else:
        best_model = fusion_model_3
        best_results = results_3
        best_y_pred = y_pred_3
        best_y_proba = y_proba_3

    results_df = pd.DataFrame([best_results])
    results_df.to_csv("feature_fusion_best_results.csv", index=False)

    training_stats = best_model.get_training_stats()
    training_stats.to_csv("feature_fusion_training_stats.csv", index=False)

    # Generate visualizations
    from vis import plot_bert_confusion_matrix, plot_bert_roc_pr
    plot_bert_confusion_matrix(y_test, best_y_pred, model_name=f'Feature Fusion - {best_experiment["experiment"]}')
    plot_bert_roc_pr(y_test, best_y_proba, model_name=f'Feature Fusion - {best_experiment["experiment"]}')

    # Optional: Save best model
    # best_model.save_model("feature_fusion_model_best")

    return comparison_df, best_model, X_text_test, X_feat_test, y_test, best_y_pred, best_y_proba


def main():
    """Main execution function"""

    # Run classical models if configured
    if RUN_CLASSICAL:
        classical_results, trainer, X_test, y_test = run_classical_models()

    # Run BERT if configured
    if RUN_BERT:
        comparison_df, best_model, X_test_text, y_test_bert, y_pred, y_proba = run_bert_model()

    # Run Feature Fusion if configured
    if RUN_FEATURE_FUSION:
        comparison_df, best_model, X_text_test, X_feat_test, y_test_fusion, y_pred, y_proba = run_feature_fusion_model()

    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()






# import pandas as pd
# from data_loader import load_data
# from feature_eng import feature_creating
# from feature_eng import mask_group_names, mask_location_names
# from train_test_split import t_t_s
# from under_over import undersample_train
# from models import ModelTrainer
# from vis import *
#
# # ============================================================================
# # CONFIGURATION: Choose what to run
# # ============================================================================
# RUN_CLASSICAL = False
# RUN_BERT = True
#
#
# # ============================================================================
#
#
# def run_classical_models():
#     """Run classical ML models (Random Forest, XGBoost, MLP)"""
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
#     # 5. Train models (with in-CV oversampling)
#     trainer = ModelTrainer()
#     trainer.fit_all(
#         X_train_und, y_train_und,
#         models=['rfc', 'xgb', 'mlpc'],
#         cv=5,
#         scoring='f1_macro',
#         n_jobs=-1
#     )
#
#     # 6. Evaluate on untouched test set
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
#     return results_df, trainer, X_test, y_test
#
#
# def run_bert_model():
#     """Run BERT model on text data with improved minority class handling"""
#     print("\n" + "=" * 80)
#     print("RUNNING BERT MODEL")
#     print("=" * 80 + "\n")
#
#     # Import improved BERT classifier
#     from non_classical import BERTClassifier
#
#     # 1. Load data
#     df = load_data()
#
#     # 2. Feature engineering (no embeddings needed for BERT)
#     working_df, unattrib_df = feature_creating(
#         df,
#         use_embeddings=False,  # BERT handles its own text encoding
#         text_columns=None
#     )
#
#     # 3. Get the text column and labels
#     X_text = working_df['notes'].fillna('')
#     print("\nRemoving group names and locations from text...")
#     X_text = X_text.apply(mask_group_names)
#     X_text = X_text.apply(mask_location_names)
#     print("Text masking complete!")
#
#     print("\n=== CHECKING FOR LEAKAGE ===")
#     sample_texts = X_text.head(10)
#     for i, text in enumerate(sample_texts):
#         has_taliban = any(word in text.lower() for word in ['taliban', 'taleban'])
#         has_isis = any(word in text.lower() for word in ['isis', 'islamic state'])
#         print(f"Text {i}: Taliban={has_taliban}, ISIS={has_isis}")
#         if has_taliban or has_isis:
#             print(f" LEAKAGE DETECTED: {text[:100]}...")
#     print("=== END CHECK ===\n")
#
#     y = working_df['target']
#
#     # 4. Train-test split for text data
#     from sklearn.model_selection import train_test_split
#     X_train_text, X_test_text, y_train, y_test = train_test_split(
#         X_text, y,
#         test_size=0.4,
#         random_state=42,
#         stratify=y
#     )
#
#     print(f"\nTraining samples: {len(X_train_text)}")
#     print(f"Test samples: {len(X_test_text)}")
#     print(f"Class distribution (train): {y_train.value_counts().to_dict()}")
#     print(f"Class distribution (test): {y_test.value_counts().to_dict()}")
#
#     # ============================================================================
#     # EXPERIMENT 1: Balanced Sampler + Moderate Class Weights
#     # ============================================================================
#     print("\n" + "=" * 80)
#     print("EXPERIMENT 1: Balanced Sampler + Moderate Class Weights")
#     print("=" * 80)
#
#     bert_model_1 = BERTClassifier(
#         model_name='bert-base-uncased',
#         max_length=256,
#         batch_size=4,
#         learning_rate=2e-5,
#         epochs=5,
#         random_state=42,
#         use_class_weights=True,  # Use moderate effective number weights
#         use_balanced_sampler=True,  # Balance batches
#         focal_loss=False
#     )
#
#     # Split train into train/val with larger validation set
#     X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
#         X_train_text, y_train,
#         test_size=0.2,  # Larger validation set
#         random_state=42,
#         stratify=y_train
#     )
#
#     bert_model_1.fit(X_train_split, y_train_split, X_val_split, y_val_split)
#
#     # Find optimal threshold on validation set
#     bert_model_1.find_optimal_threshold(X_val_split, y_val_split)
#
#     results_1, y_pred_1, y_proba_1 = bert_model_1.evaluate(X_test_text, y_test)
#
#     # ============================================================================
#     # EXPERIMENT 2: Focal Loss (no class weights)
#     # ============================================================================
#     print("\n" + "=" * 80)
#     print("EXPERIMENT 2: Focal Loss")
#     print("=" * 80)
#
#     bert_model_2 = BERTClassifier(
#         model_name='bert-base-uncased',
#         max_length=256,
#         batch_size=4,
#         learning_rate=2e-5,
#         epochs=5,
#         random_state=42,
#         use_class_weights=False,
#         use_balanced_sampler=True,
#         focal_loss=True,  # Use focal loss
#         focal_alpha=0.25,
#         focal_gamma=2.0
#     )
#
#     bert_model_2.fit(X_train_split, y_train_split, X_val_split, y_val_split)
#     bert_model_2.find_optimal_threshold(X_val_split, y_val_split)
#     results_2, y_pred_2, y_proba_2 = bert_model_2.evaluate(X_test_text, y_test)
#
#     # ============================================================================
#     # EXPERIMENT 3: Balanced Sampler Only (baseline)
#     # ============================================================================
#     print("\n" + "=" * 80)
#     print("EXPERIMENT 3: Balanced Sampler Only (baseline)")
#     print("=" * 80)
#
#     bert_model_3 = BERTClassifier(
#         model_name='bert-base-uncased',
#         max_length=256,
#         batch_size=4,
#         learning_rate=2e-5,
#         epochs=5,
#         random_state=42,
#         use_class_weights=False,
#         use_balanced_sampler=True,
#         focal_loss=False
#     )
#
#     bert_model_3.fit(X_train_split, y_train_split, X_val_split, y_val_split)
#     bert_model_3.find_optimal_threshold(X_val_split, y_val_split)
#     results_3, y_pred_3, y_proba_3 = bert_model_3.evaluate(X_test_text, y_test)
#
#     # ============================================================================
#     # COMPARE RESULTS
#     # ============================================================================
#     print("\n" + "=" * 80)
#     print("COMPARING ALL EXPERIMENTS")
#     print("=" * 80)
#
#     comparison_df = pd.DataFrame([
#         {
#             'experiment': 'Balanced Sampler + Class Weights',
#             'macro_f1': results_1['test_f1_macro'],
#             'accuracy': results_1['test_accuracy'],
#             'minority_recall': results_1['minority_recall'],
#             'minority_precision': results_1['minority_precision'],
#             'minority_f1': results_1['minority_f1']
#         },
#         {
#             'experiment': 'Focal Loss',
#             'macro_f1': results_2['test_f1_macro'],
#             'accuracy': results_2['test_accuracy'],
#             'minority_recall': results_2['minority_recall'],
#             'minority_precision': results_2['minority_precision'],
#             'minority_f1': results_2['minority_f1']
#         },
#         {
#             'experiment': 'Balanced Sampler Only',
#             'macro_f1': results_3['test_f1_macro'],
#             'accuracy': results_3['test_accuracy'],
#             'minority_recall': results_3['minority_recall'],
#             'minority_precision': results_3['minority_precision'],
#             'minority_f1': results_3['minority_f1']
#         }
#     ])
#
#     print(comparison_df.to_string(index=False))
#
#     # Find best approach
#     best_idx = comparison_df['minority_f1'].idxmax()
#     best_experiment = comparison_df.iloc[best_idx]
#
#     print("\n" + "=" * 80)
#     print(f"BEST APPROACH: {best_experiment['experiment']}")
#     print("=" * 80)
#     print(f"Minority F1: {best_experiment['minority_f1']:.4f}")
#     print(f"Minority Recall: {best_experiment['minority_recall']:.4f}")
#     print(f"Minority Precision: {best_experiment['minority_precision']:.4f}")
#     print(f"Macro F1: {best_experiment['macro_f1']:.4f}")
#
#     # Save results
#     comparison_df.to_csv("bert_experiments_comparison.csv", index=False)
#     print("\nComparison saved to 'bert_experiments_comparison.csv'")
#
#     # Save best model's detailed results
#     if best_idx == 0:
#         best_model = bert_model_1
#         best_results = results_1
#         best_y_pred = y_pred_1
#         best_y_proba = y_proba_1
#     elif best_idx == 1:
#         best_model = bert_model_2
#         best_results = results_2
#         best_y_pred = y_pred_2
#         best_y_proba = y_proba_2
#     else:
#         best_model = bert_model_3
#         best_results = results_3
#         best_y_pred = y_pred_3
#         best_y_proba = y_proba_3
#
#     results_df = pd.DataFrame([best_results])
#     results_df.to_csv("bert_best_results.csv", index=False)
#
#     training_stats = best_model.get_training_stats()
#     training_stats.to_csv("bert_best_training_stats.csv", index=False)
#
#     # Generate visualizations
#     from vis import plot_bert_confusion_matrix, plot_bert_roc_pr
#     plot_bert_confusion_matrix(y_test, best_y_pred, model_name=f'BERT - {best_experiment["experiment"]}')
#     plot_bert_roc_pr(y_test, best_y_proba, model_name=f'BERT - {best_experiment["experiment"]}')
#
#     # Optional: Save best model
#     # best_model.save_model("bert_model_best")
#
#     return comparison_df, best_model, X_test_text, y_test, best_y_pred, best_y_proba
#
#
# def main():
#     """Main execution function"""
#
#     # Run classical models if configured
#     if RUN_CLASSICAL:
#         classical_results, trainer, X_test, y_test = run_classical_models()
#
#     # Run BERT if configured
#     if RUN_BERT:
#         comparison_df, best_model, X_test_text, y_test_bert, y_pred, y_proba = run_bert_model()
#
#     print("\n" + "=" * 80)
#     print("ALL TASKS COMPLETED")
#     print("=" * 80)
#
#
# if __name__ == "__main__":
#     main()