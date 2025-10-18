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
RUN_BERT = True


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
    """Run BERT model on text data"""
    print("\n" + "=" * 80)
    print("RUNNING BERT MODEL")
    print("=" * 80 + "\n")

    # Import BERT classifier
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
    # Extract 'notes' column before train-test split
    X_text = working_df['notes'].fillna('')  # Handle missing values
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

    # 5. Initialize and train BERT
    bert_model = BERTClassifier(
        model_name='bert-base-uncased',
        max_length=128,
        batch_size=8,
        learning_rate=2e-5,
        epochs=3,
        random_state=42
    )

    # Optional: Split train into train/val for validation during training
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_text, y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )

    # Train with validation set
    bert_model.fit(X_train_split, y_train_split, X_val_split, y_val_split)

    # 6. Evaluate on test set
    results, y_pred, y_proba = bert_model.evaluate(X_test_text, y_test)

    # 7. Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv("bert_results.csv", index=False)
    print("\nResults saved to 'bert_results.csv'")

    # 8. Save training stats
    training_stats = bert_model.get_training_stats()
    training_stats.to_csv("bert_training_stats.csv", index=False)
    print("Training stats saved to 'bert_training_stats.csv'")

    # 9. Generate BERT visualizations
    from vis import plot_bert_confusion_matrix, plot_bert_roc_pr

    plot_bert_confusion_matrix(y_test, y_pred, model_name='BERT')
    plot_bert_roc_pr(y_test, y_proba, model_name='BERT')

    # 10. Optional: Save model
    # bert_model.save_model("bert_model_saved")

    return results_df, bert_model, X_test_text, y_test, y_pred, y_proba


def main():
    """Main execution function"""

    # Run classical models if configured
    if RUN_CLASSICAL:
        classical_results, trainer, X_test, y_test = run_classical_models()

    # Run BERT if configured
    if RUN_BERT:
        bert_results, bert_model, X_test_text, y_test_bert, y_pred, y_proba = run_bert_model()

    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()



# import pandas as pd
# from data_loader import load_data
# from feature_eng import feature_creating
# from train_test_split import t_t_s
# from under_over import undersample_train
# from models import ModelTrainer
# from vis import *
#
#
#
#
#
# # 1. Load data
# df = load_data()
#
# # 2. Feature engineering (includes embeddings)
# text_columns_to_embed = ['notes']
# working_df, unattrib_df = feature_creating(df, use_embeddings=True, text_columns=text_columns_to_embed)
#
# # 3. Train-test split
# X_train, X_test, y_train, y_test = t_t_s(working_df)
#
# # 4. Undersample training set only
# X_train_und, y_train_und = undersample_train(X_train, y_train, final_majority_size=2500)
#
# # 5. Train models (with in-CV oversampling)
# trainer = ModelTrainer()
# trainer.fit_all(X_train_und, y_train_und, models=['rfc', 'xgb', 'mlpc'], cv=5, scoring='f1_macro', n_jobs=-1)
#
# # 6. Evaluate on untouched test set
# trainer.evaluate(X_test, y_test)
#
# # 7. Results summary
# results_df = trainer.get_results()
# print(results_df[['model', 'cv_score', 'f1_macro', 'accuracy', 'precision', 'recall']])
#
# # 8. Best model
# best_model, preprocessors, best_stats = trainer.get_best_model(metric='f1_macro')
# print("\nBest Model:")
# print(best_stats)
#
# # 9. Save results
# results_df.to_csv("oversample_comparison_results.csv", index=False)
# print("Results saved to 'oversample_comparison_results.csv'")
#
# plot_model_performance(results_df, metric='f1_macro')
# plot_confusion_matrix_best(trainer, X_test, y_test)
# plot_roc_pr(trainer, X_test, y_test)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # import pandas as pd
# # from data_loader import load_data
# # from feature_eng import feature_creating
# # from train_test_split import t_t_s
# # from under_over import undersample_train
# # from models import ModelTrainer
# #
# #
# #
# #
# # ###---LOAD DATA---###
# # df = load_data()
# # ###---LOAD DATA---###
# #
# # ###---get embeddings----
# #
# # ###---get embeddings----
# #
# # ###---PRE-PROCESS + FEATURES---###
# # #working_df, unattrib_df  = feature_creating(df)
# # text_columns_to_embed = ['notes']  # Add more columns if needed: ['notes', 'actor2', 'tags']
# # working_df, unattrib_df = feature_creating(df, use_embeddings=True, text_columns=text_columns_to_embed)
# # print(working_df.shape)
# # print(working_df['target'].value_counts())
# # print(working_df['target'].unique())
# # print(list(working_df.columns))
# # ###---PRE-PROCESS + FEATURES---###
# #
# # #---Train Test Split---
# # X_train, X_test, y_train, y_test = t_t_s(working_df)
# #
# # print(X_train.shape)
# # print(y_train.shape)
# # #---Train Test Split---
# #
# #
# #
# # #---Under Sample---
# # X_train_und, y_train_und = undersample_train(X_train, y_train, final_majority_size=2500)
# # print(X_train_und.shape)
# # print(y_train_und.shape)
# # #---Under Sample---
# #
# #
# #
# #
# #
# #
# # #---SMOTE ONLY---
# # # X_bal, y_bal = smote(X_train_und, y_train_und)
# # # print(X_bal.shape)
# # # print(y_bal.shape)
# # #---SMOTE ONLY---
# #
# # #---All Oversample---
# # oversampled_datasets = ensamble_oversamp(X_train_und, y_train_und,
# #                                          random_seed=42,
# #                                          sampling_strategy='minority')
# # #---All Oversample---
# #
# #
# #
# #
# #
# # ###---CLASSICAL MODELS---###
# # all_results = []
# #
# # for method_name, (X_bal, y_bal) in oversampled_datasets.items():
# #     print("\n" + "=" * 80)
# #     print(f"TRAINING MODELS WITH {method_name.upper()} OVERSAMPLING")
# #     print("=" * 80)
# #     print(f"Dataset shape: {X_bal.shape}")
# #     print(f"Class distribution: {pd.Series(y_bal).value_counts().to_dict()}")
# #
# #     # Initialize a new trainer for each oversampling method
# #     trainer = ModelTrainer()
# #
# #     # Train both RF and XGBoost models
# #     trainer.fit_all(X_bal, y_bal, models=['rfc', 'xgb', 'mlpc'], cv=5, scoring='f1_macro', n_jobs=-1)
# #
# #     # Evaluate on test set
# #     trainer.evaluate(X_test, y_test)
# #
# #     # Get results and add method name
# #     results_df = trainer.get_results()
# #     results_df['oversample_method'] = method_name
# #
# #     # Store results
# #     all_results.append(results_df)
# #
# #     # Display results for this method
# #     print(f"\nResults for {method_name}:")
# #     print(results_df[['model', 'cv_score', 'f1_macro', 'accuracy', 'precision', 'recall']])
# #
# # ###---TRAIN MODELS ON EACH OVERSAMPLED DATASET---###
# #
# #
# # ###---COMPARE ALL RESULTS---###
# # # Combine all results
# # combined_results = pd.concat(all_results, ignore_index=True)
# #
# # print("\n" + "=" * 80)
# # print("COMPLETE RESULTS COMPARISON")
# # print("=" * 80)
# # print(combined_results[['oversample_method', 'model', 'cv_score', 'f1_macro',
# #                         'accuracy', 'precision', 'recall']].to_string())
# #
# # # Find best overall combination
# # best_idx = combined_results['f1_macro'].idxmax()
# # best_result = combined_results.loc[best_idx]
# #
# # print("\n" + "=" * 80)
# # print("BEST OVERALL COMBINATION")
# # print("=" * 80)
# # print(f"Oversampling Method: {best_result['oversample_method'].upper()}")
# # print(f"Model: {best_result['model'].upper()}")
# # print(f"Test F1 Score: {best_result['f1_macro']:.4f}")
# # print(f"Test Accuracy: {best_result['accuracy']:.4f}")
# # print(f"CV Score: {best_result['cv_score']:.4f}")
# # print(f"Precision: {best_result['precision']:.4f}")
# # print(f"Recall: {best_result['recall']:.4f}")
# #
# # # Save results to CSV for later analysis
# # combined_results.to_csv('oversample_comparison_results.csv', index=False)
# # print("\nResults saved to 'oversample_comparison_results.csv'")
# # ###---COMPARE ALL RESULTS---###
# #
# # ###---CLASSICAL MODELS---###
# #
# #
