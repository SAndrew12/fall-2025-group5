"""
eXplainable AI (XAI) Module
Compatible with your exact ModelTrainer, BERTClassifier, and Feature Fusion models

Usage:
    from xai_explanations import explain_classical_models, explain_bert_model, quick_xai

    # After training classical models
    explain_classical_models(trainer, X_train, X_test, y_test, feature_names)

    # Or for quick summary
    quick_xai(trainer, X_train, X_test, y_test, feature_names)

    # After training BERT
    explain_bert_model(bert_model, X_test_text, y_test)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')

# Create output directory
XAI_DIR = 'xai_explanations'
if not os.path.exists(XAI_DIR):
    os.makedirs(XAI_DIR)


# ============================================================================
# CLASSICAL MODEL EXPLANATIONS
# ============================================================================

def plot_feature_importance_classical(model, feature_names, model_name='Model',
                                      top_n=20, save_dir=XAI_DIR):
    """Plot feature importance for tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        print(f"{model_name} does not have feature_importances_")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Features: {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'feature_importance_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    return pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })


def explain_with_shap_classical(model, X_train, X_test, feature_names,
                                model_name='Model', max_display=20, save_dir=XAI_DIR):
    """Generate SHAP explanations for tree-based models"""
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap --break-system-packages")
        return None, None

    print(f"Generating SHAP explanations for {model_name}...")

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, get positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.title(f'SHAP Summary: {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'shap_summary_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type='bar', max_display=max_display, show=False)
    plt.title(f'SHAP Importance: {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'shap_bar_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    return shap_values, explainer


def explain_with_lime_classical(model, X_train, X_test, feature_names,
                                sample_idx=0, num_features=20,
                                model_name='Model', save_dir=XAI_DIR):
    """Generate LIME explanation for a single prediction"""
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        print("LIME not installed. Run: pip install lime --break-system-packages")
        return None

    print(f"Generating LIME explanation for {model_name} (sample {sample_idx})...")

    # Convert to numpy if needed
    X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    # Create explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_np,
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
        mode='classification'
    )

    # Explain instance
    if hasattr(model, 'predict_proba'):
        explanation = explainer.explain_instance(
            X_test_np[sample_idx],
            model.predict_proba,
            num_features=num_features
        )
    else:
        def predict_wrapper(X):
            preds = model.predict(X)
            return np.column_stack([1 - preds, preds])

        explanation = explainer.explain_instance(
            X_test_np[sample_idx],
            predict_wrapper,
            num_features=num_features
        )

    # Save plot
    fig = explanation.as_pyplot_figure()
    plt.title(f'LIME - Sample {sample_idx}: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'lime_sample_{sample_idx}_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    return explanation


def explain_classical_models(trainer, X_train, X_test, y_test, feature_names,
                             sample_indices=[0, 1, 2], save_dir=XAI_DIR):
    """
    Comprehensive XAI for classical models trained with ModelTrainer

    Args:
        trainer: Your ModelTrainer instance (after fit_all() and evaluate())
        X_train: Training features (original, not preprocessed)
        X_test: Test features (original, not preprocessed)
        y_test: Test labels
        feature_names: List of feature names
        sample_indices: Indices of samples to explain with LIME
        save_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("EXPLAINABLE AI - CLASSICAL MODELS")
    print("=" * 80)

    # Get best model
    best_model, preprocessors, best_stats = trainer.get_best_model(metric='test_f1_macro')
    model_name = best_stats['model'].replace('_', ' ').title()

    print(f"\nBest Model: {model_name}")
    print(f"Test F1 Macro: {best_stats['test_f1_macro']:.4f}")

    # Preprocess data (same way trainer does it)
    X_train_proc = trainer._apply_preprocessing(X_train, preprocessors, fit=False)
    X_test_proc = trainer._apply_preprocessing(X_test, preprocessors, fit=False)

    # 1. Feature Importance
    print("\n--- Feature Importance ---")
    importance_df = plot_feature_importance_classical(
        best_model, feature_names, model_name=model_name, top_n=20, save_dir=save_dir
    )
    if importance_df is not None:
        print("\nTop 10 Features:")
        print(importance_df.head(10).to_string(index=False))

    # 2. SHAP Explanations (if tree-based)
    model_type = type(best_model).__name__.lower()
    if any(x in model_type for x in ['forest', 'xgb', 'gradient', 'tree']):
        print("\n--- SHAP Explanations ---")
        shap_values, explainer = explain_with_shap_classical(
            best_model, X_train_proc, X_test_proc, feature_names,
            model_name=model_name, max_display=20, save_dir=save_dir
        )
    else:
        print(f"\n{model_name} is not tree-based, skipping SHAP (use LIME instead)")
        shap_values, explainer = None, None

    # 3. LIME Explanations
    print("\n--- LIME Explanations ---")
    for idx in sample_indices[:3]:
        if idx < len(X_test_proc):
            explain_with_lime_classical(
                best_model, X_train_proc, X_test_proc, feature_names,
                sample_idx=idx, num_features=15,
                model_name=model_name, save_dir=save_dir
            )

    print("\n" + "=" * 80)
    print(f"XAI outputs saved to: {save_dir}/")
    print("=" * 80 + "\n")

    return importance_df, shap_values, explainer


# ============================================================================
# BERT MODEL EXPLANATIONS
# ============================================================================

def explain_with_lime_text(bert_model, texts, sample_idx=0, num_features=20,
                           model_name='BERT', save_dir=XAI_DIR):
    """Generate LIME text explanation for BERT model"""
    try:
        import lime
        import lime.lime_text
    except ImportError:
        print("LIME not installed. Run: pip install lime --break-system-packages")
        return None

    print(f"Generating LIME text explanation (sample {sample_idx})...")

    # Get text
    text = texts.iloc[sample_idx] if isinstance(texts, pd.Series) else texts[sample_idx]

    # Create explainer
    explainer = lime.lime_text.LimeTextExplainer(class_names=['Class 0', 'Class 1'])

    # Explain
    explanation = explainer.explain_instance(
        text,
        bert_model.predict_proba,
        num_features=num_features
    )

    # Save plot
    fig = explanation.as_pyplot_figure()
    plt.title(f'LIME Text - Sample {sample_idx}: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'lime_text_sample_{sample_idx}_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Save HTML
    html_path = os.path.join(save_dir, f'lime_text_sample_{sample_idx}_{model_name}.html')
    explanation.save_to_file(html_path)
    print(f"Saved HTML: {html_path}")

    return explanation


def visualize_bert_attention(bert_model, text, layer=-1, head=0,
                             sample_idx=0, save_dir=XAI_DIR):
    """Visualize BERT attention weights"""
    try:
        from transformers import AutoTokenizer
        import torch
    except ImportError:
        print("Transformers not installed")
        return None

    print(f"Visualizing BERT attention (sample {sample_idx})...")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(bert_model.model_name)
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       max_length=bert_model.max_length, padding=True)

    # Move to device
    inputs = {k: v.to(bert_model.device) for k, v in inputs.items()}

    # Get attention
    with torch.no_grad():
        outputs = bert_model.model(**inputs, output_attentions=True)
        attentions = outputs.attentions

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attention = attentions[layer][0, head].cpu().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attention, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'BERT Attention - Layer {layer}, Head {head}\nSample {sample_idx}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Tokens', fontsize=11)
    ax.set_ylabel('Source Tokens', fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'bert_attention_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    return attention, tokens


def explain_bert_model(bert_model, X_test_text, y_test, sample_indices=[0, 1, 2],
                       model_name='BERT', save_dir=XAI_DIR):
    """
    Comprehensive XAI for BERT model

    Args:
        bert_model: Your BERTClassifier instance (after fit())
        X_test_text: Test text samples
        y_test: Test labels
        sample_indices: Indices to explain in detail
        model_name: Name for outputs
        save_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print(f"EXPLAINABLE AI - {model_name}")
    print("=" * 80)

    # 1. LIME Text Explanations
    print("\n--- LIME Text Explanations ---")
    for idx in sample_indices[:3]:
        if idx < len(X_test_text):
            explain_with_lime_text(
                bert_model, X_test_text, sample_idx=idx,
                num_features=20, model_name=model_name, save_dir=save_dir
            )

    # 2. Attention Visualization
    print("\n--- BERT Attention Visualization ---")
    for idx in sample_indices[:2]:  # Limit to 2 (can be slow)
        if idx < len(X_test_text):
            text = X_test_text.iloc[idx] if isinstance(X_test_text, pd.Series) else X_test_text[idx]
            visualize_bert_attention(
                bert_model, text, layer=-1, head=0,
                sample_idx=idx, save_dir=save_dir
            )

    print("\n" + "=" * 80)
    print(f"XAI outputs saved to: {save_dir}/")
    print("=" * 80 + "\n")


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def analyze_errors_classical(trainer, X_train, X_test, y_test, feature_names,
                             n_errors=5, save_dir=XAI_DIR):
    """
    Analyze misclassified samples with XAI

    Args:
        trainer: ModelTrainer instance
        X_train: Training features
        X_test: Test features
        y_test: Test labels
        feature_names: Feature names
        n_errors: Number of errors to analyze
        save_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS WITH XAI")
    print("=" * 80)

    # Get best model
    best_model, preprocessors, best_stats = trainer.get_best_model(metric='test_f1_macro')
    model_name = best_stats['model'].replace('_', ' ').title()

    # Preprocess
    X_train_proc = trainer._apply_preprocessing(X_train, preprocessors, fit=False)
    X_test_proc = trainer._apply_preprocessing(X_test, preprocessors, fit=False)

    # Get predictions
    y_pred = best_model.predict(X_test_proc)

    # Find errors
    error_mask = y_test.values != y_pred if hasattr(y_test, 'values') else y_test != y_pred
    error_indices = np.where(error_mask)[0]

    print(f"\nTotal errors: {len(error_indices)}")
    print(f"Analyzing first {min(n_errors, len(error_indices))} errors...")

    # Explain each error
    for i, idx in enumerate(error_indices[:n_errors]):
        true_label = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        pred_label = y_pred[idx]

        print(f"\nError {i + 1}: True={true_label}, Predicted={pred_label}")

        explain_with_lime_classical(
            best_model, X_train_proc, X_test_proc, feature_names,
            sample_idx=idx, num_features=15,
            model_name=f"{model_name}_error_{i}", save_dir=save_dir
        )

    print("\n" + "=" * 80)
    print(f"Error analysis saved to: {save_dir}/")
    print("=" * 80 + "\n")


# ============================================================================
# QUICK XAI - CONVENIENCE FUNCTION
# ============================================================================

def quick_xai(trainer, X_train, X_test, y_test, feature_names, save_dir=XAI_DIR):
    """
    Quick XAI summary - essential explanations only

    Args:
        trainer: ModelTrainer instance
        X_train: Training features
        X_test: Test features
        y_test: Test labels
        feature_names: Feature names
        save_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("QUICK XAI SUMMARY")
    print("=" * 80)

    # Get best model
    best_model, preprocessors, best_stats = trainer.get_best_model(metric='test_f1_macro')
    model_name = best_stats['model'].replace('_', ' ').title()

    # Preprocess
    X_train_proc = trainer._apply_preprocessing(X_train, preprocessors, fit=False)
    X_test_proc = trainer._apply_preprocessing(X_test, preprocessors, fit=False)

    # Feature importance
    plot_feature_importance_classical(
        best_model, feature_names, model_name=model_name, top_n=15, save_dir=save_dir
    )

    # SHAP if tree-based
    model_type = type(best_model).__name__.lower()
    if any(x in model_type for x in ['forest', 'xgb', 'gradient', 'tree']):
        explain_with_shap_classical(
            best_model, X_train_proc, X_test_proc, feature_names,
            model_name=model_name, max_display=15, save_dir=save_dir
        )

    # One LIME example
    explain_with_lime_classical(
        best_model, X_train_proc, X_test_proc, feature_names,
        sample_idx=0, num_features=15,
        model_name=model_name, save_dir=save_dir
    )

    print("\n" + "=" * 80)
    print(f"Quick XAI complete! Check: {save_dir}/")
    print("=" * 80 + "\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\nXAI Module Loaded Successfully!")
    print("\nTo install required packages:")
    print("  pip install shap lime --break-system-packages")