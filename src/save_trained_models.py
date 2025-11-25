"""
Script to Save Trained Models for Streamlit Demo
This script shows you how to save all three types of models after training
"""

import os
import pickle
import joblib
import json
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd


# ============================================================================
# 1. SAVING CLASSICAL MODELS (Random Forest, XGBoost, MLP)
# ============================================================================

def save_classical_models(trainer, feature_names, model_dir='saved_models/classical'):
    """
    Save classical ML models with their preprocessors

    Args:
        trainer: Your ModelTrainer instance after training
        feature_names: List of feature names used in training
        model_dir: Directory to save models
    """
    os.makedirs(model_dir, exist_ok=True)

    # Get the best model and its preprocessors
    best_model, preprocessors, best_stats = trainer.get_best_model(metric='test_f1_macro')

    print(f"\n{'=' * 60}")
    print("SAVING CLASSICAL MODELS")
    print(f"{'=' * 60}")

    # Save the best model
    joblib.dump(best_model, f'{model_dir}/best_model.pkl')
    print(f"✓ Best model saved: {best_stats['model']}")

    # Save all trained models (optional - if you want all models available)
    for model_name, model in trainer.trained_models.items():
        joblib.dump(model, f'{model_dir}/{model_name}.pkl')
    print(f"✓ All {len(trainer.trained_models)} models saved")

    # Save preprocessors (scalers, etc.)
    joblib.dump(preprocessors, f'{model_dir}/preprocessors.pkl')
    print("✓ Preprocessors saved")

    # Save feature names
    with open(f'{model_dir}/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("✓ Feature names saved")

    # Save model metadata
    metadata = {
        'best_model_name': best_stats['model'],
        'test_f1_macro': float(best_stats['test_f1_macro']),
        'test_accuracy': float(best_stats['test_accuracy']),
        'test_precision': float(best_stats['test_precision']),
        'test_recall': float(best_stats['test_recall']),
        'num_features': len(feature_names)
    }

    with open(f'{model_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✓ Metadata saved")

    print(f"\nClassical models saved to: {model_dir}/")
    print(f"{'=' * 60}\n")

    return model_dir


# ============================================================================
# 2. SAVING BERT MODEL
# ============================================================================

def save_bert_model(bert_model, model_dir='saved_models/bert'):
    """
    Save BERT model

    Args:
        bert_model: Your trained BERTClassifier instance
        model_dir: Directory to save model
    """
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("SAVING BERT MODEL")
    print(f"{'=' * 60}")

    # Use the built-in save method
    bert_model.save_model(model_dir)

    # Save additional config information
    config = {
        'max_length': bert_model.max_length,
        'batch_size': bert_model.batch_size,
        'prediction_threshold': bert_model.prediction_threshold,
        'model_name': bert_model.model_name,
        'use_focal_loss': bert_model.use_focal_loss,
        'focal_alpha': bert_model.focal_alpha,
        'focal_gamma': bert_model.focal_gamma
    }

    with open(f'{model_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Configuration saved")

    print(f"\nBERT model saved to: {model_dir}/")
    print(f"{'=' * 60}\n")

    return model_dir


# ============================================================================
# 3. SAVING FEATURE FUSION MODEL
# ============================================================================

def save_feature_fusion_model(fusion_model, feature_names, scaler,
                              model_dir='saved_models/feature_fusion'):
    """
    Save Feature Fusion model with all necessary components

    Args:
        fusion_model: Your trained BERTFeatureFusionClassifier instance
        feature_names: List of manual feature names used
        scaler: The StandardScaler fitted on manual features
        model_dir: Directory to save model
    """
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("SAVING FEATURE FUSION MODEL")
    print(f"{'=' * 60}")

    # Use the built-in save method for the model
    fusion_model.save_model(model_dir)

    # Save the scaler used for manual features
    joblib.dump(scaler, f'{model_dir}/feature_scaler.pkl')
    print("✓ Feature scaler saved")

    # Save manual feature names
    with open(f'{model_dir}/manual_feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("✓ Manual feature names saved")

    # Save configuration
    config = {
        'max_length': fusion_model.max_length,
        'batch_size': fusion_model.batch_size,
        'prediction_threshold': fusion_model.prediction_threshold,
        'model_name': fusion_model.model_name,
        'num_manual_features': fusion_model.num_manual_features,
        'hidden_dim': fusion_model.hidden_dim,
        'use_focal_loss': fusion_model.use_focal_loss,
        'focal_alpha': fusion_model.focal_alpha,
        'focal_gamma': fusion_model.focal_gamma,
        'manual_features': feature_names
    }

    with open(f'{model_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Configuration saved")

    print(f"\nFeature Fusion model saved to: {model_dir}/")
    print(f"{'=' * 60}\n")

    return model_dir


# ============================================================================
# 4. COMPLETE WORKFLOW EXAMPLE
# ============================================================================

def save_all_models_workflow():
    """
    Complete example showing how to save models after training
    This is what you would add to your main.py
    """
    print("\n" + "=" * 80)
    print("COMPLETE MODEL SAVING WORKFLOW")
    print("=" * 80 + "\n")

    # Example: After running classical models
    print("After running run_classical_models():")
    print("-" * 60)
    print("""
    results_df, trainer, X_test, y_test = run_classical_models()

    # Save classical models
    feature_names = X_test.columns.tolist()
    save_classical_models(trainer, feature_names)
    """)

    # Example: After running BERT
    print("\nAfter running run_bert_model():")
    print("-" * 60)
    print("""
    results_df, bert_model, X_test_text, y_test_bert, y_pred, y_proba = run_bert_model()

    # Save BERT model
    save_bert_model(bert_model)
    """)

    # Example: After running Feature Fusion
    print("\nAfter running run_feature_fusion():")
    print("-" * 60)
    print("""
    results_df, fusion_model, X_text_test, X_man_test, y_test_fusion, y_pred, y_proba = run_feature_fusion()

    # Save feature fusion model (remember to save the scaler!)
    manual_feature_names = X_man_test.columns.tolist()
    save_feature_fusion_model(fusion_model, manual_feature_names, scaler)
    """)

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    print("Model Saving Utilities Loaded!")
    print("\nThis script provides functions to save:")
    print("  1. Classical ML models (RF, XGB, MLP)")
    print("  2. BERT model")
    print("  3. Feature Fusion model")
    print("\nSee save_all_models_workflow() for usage examples")