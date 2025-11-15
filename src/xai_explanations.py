"""
Enhanced eXplainable AI (XAI) Module
Provides GLOBAL feature importance analysis across ALL models

Key improvements:
1. Feature importance for ALL tree-based models (not just best)
2. Global aggregated SHAP values across all models
3. Comparative feature importance visualization
4. Summary statistics and consensus features
5. Optional LIME for specific sample analysis

Usage:
    from xai_explanations_enhanced import explain_all_models, compare_feature_importance

    # After training and evaluation
    explain_all_models(trainer, X_train, X_test, y_test, feature_names)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, List, Dict, Tuple
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Create output directory
XAI_DIR = 'xai_explanations'
if not os.path.exists(XAI_DIR):
    os.makedirs(XAI_DIR)


# ============================================================================
# GLOBAL FEATURE IMPORTANCE - ALL MODELS
# ============================================================================

def extract_all_feature_importances(trainer, feature_names, save_dir=XAI_DIR):
    """
    Extract feature importance from ALL tree-based models

    Returns:
        DataFrame with feature importances from all models
    """
    print("\n" + "=" * 80)
    print("EXTRACTING FEATURE IMPORTANCE FROM ALL TREE-BASED MODELS")
    print("=" * 80)

    importance_data = []

    # Iterate through all trained models
    for model_key, (model, preprocessors) in trainer.trained_models.items():
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            # Store as a record
            for feat_name, importance in zip(feature_names, importances):
                importance_data.append({
                    'model': model_key,
                    'feature': feat_name,
                    'importance': importance
                })

            print(f"✓ Extracted from {model_key}")
        else:
            print(f"✗ Skipped {model_key} (no feature_importances_)")

    if not importance_data:
        print("\nNo tree-based models found!")
        return None

    df_importance = pd.DataFrame(importance_data)

    # Save raw data
    csv_path = os.path.join(save_dir, 'all_feature_importances.csv')
    df_importance.to_csv(csv_path, index=False)
    print(f"\n✓ Saved raw importance data to: {csv_path}")

    return df_importance


def aggregate_feature_importance(df_importance, top_n=25, save_dir=XAI_DIR):
    """
    Aggregate feature importance across all models

    Creates:
    1. Mean importance across models
    2. Std deviation (shows consistency)
    3. Frequency (how many models found it important)
    """
    print("\n" + "=" * 80)
    print("AGGREGATING FEATURE IMPORTANCE")
    print("=" * 80)

    # Calculate statistics per feature
    agg_stats = df_importance.groupby('feature').agg({
        'importance': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)

    agg_stats.columns = ['mean_importance', 'std_importance', 'min_importance',
                         'max_importance', 'model_count']
    agg_stats = agg_stats.sort_values('mean_importance', ascending=False)

    # Save aggregated stats
    csv_path = os.path.join(save_dir, 'aggregated_feature_importance.csv')
    agg_stats.to_csv(csv_path)
    print(f"\n✓ Saved aggregated importance to: {csv_path}")

    # Print top features
    print(f"\n{'=' * 80}")
    print(f"TOP {min(top_n, len(agg_stats))} MOST IMPORTANT FEATURES (Averaged Across All Models)")
    print('=' * 80)
    print(agg_stats.head(top_n).to_string())

    return agg_stats


def plot_comparative_feature_importance(df_importance, top_n=20, save_dir=XAI_DIR):
    """
    Create visualizations comparing feature importance across models
    """
    print("\n" + "=" * 80)
    print("CREATING COMPARATIVE VISUALIZATIONS")
    print("=" * 80)

    # 1. Heatmap: Features vs Models
    pivot_table = df_importance.pivot_table(
        values='importance',
        index='feature',
        columns='model',
        aggfunc='mean',
        fill_value=0
    )

    # Get top features by mean importance
    top_features = pivot_table.mean(axis=1).nlargest(top_n).index
    pivot_subset = pivot_table.loc[top_features]

    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_subset, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Importance'})
    plt.title(f'Feature Importance Heatmap: Top {top_n} Features Across All Models',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'feature_importance_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap: {save_path}")
    plt.close()

    # 2. Box plot: Distribution of importance for top features
    top_features_list = pivot_table.mean(axis=1).nlargest(15).index.tolist()
    df_top = df_importance[df_importance['feature'].isin(top_features_list)]

    plt.figure(figsize=(14, 8))
    df_top_sorted = df_top.groupby('feature')['importance'].mean().sort_values(ascending=False)
    order = df_top_sorted.index.tolist()

    sns.boxplot(data=df_top, x='importance', y='feature', order=order, palette='Set2')
    plt.title('Feature Importance Distribution Across Models (Top 15)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'feature_importance_boxplot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved boxplot: {save_path}")
    plt.close()

    # 3. Bar plot: Mean importance with error bars
    agg = df_importance.groupby('feature')['importance'].agg(['mean', 'std']).sort_values('mean', ascending=False).head(
        top_n)

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(agg)), agg['mean'].values, xerr=agg['std'].values,
             color='steelblue', alpha=0.7, capsize=5)
    plt.yticks(range(len(agg)), agg.index)
    plt.xlabel('Mean Importance ± Std', fontsize=12)
    plt.title(f'Top {top_n} Features: Mean Importance Across All Models',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'mean_feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved mean importance plot: {save_path}")
    plt.close()


def identify_consensus_features(df_importance, top_n=15, threshold=0.01):
    """
    Identify features that are consistently important across models

    Consensus = features in top N for most models OR above threshold in all models
    """
    print("\n" + "=" * 80)
    print("IDENTIFYING CONSENSUS FEATURES")
    print("=" * 80)

    consensus = []

    # For each model, get top N features
    for model_name, group in df_importance.groupby('model'):
        top_features = group.nlargest(top_n, 'importance')['feature'].tolist()
        consensus.extend(top_features)

    # Count how many times each feature appears in top N
    from collections import Counter
    feature_counts = Counter(consensus)

    # Get features that appear in top N for at least 50% of models
    num_models = df_importance['model'].nunique()
    consensus_threshold = num_models * 0.5

    consensus_features = {feat: count for feat, count in feature_counts.items()
                          if count >= consensus_threshold}

    print(f"\nFound {len(consensus_features)} consensus features")
    print(f"(Features appearing in top {top_n} for ≥50% of models)\n")

    # Create a sorted DataFrame
    consensus_df = pd.DataFrame([
        {'feature': feat, 'appearances': count, 'percentage': count / num_models * 100}
        for feat, count in sorted(consensus_features.items(), key=lambda x: x[1], reverse=True)
    ])

    print(consensus_df.to_string(index=False))

    # Save
    csv_path = os.path.join(XAI_DIR, 'consensus_features.csv')
    consensus_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved consensus features to: {csv_path}")

    return consensus_df


# ============================================================================
# GLOBAL SHAP ANALYSIS
# ============================================================================

def global_shap_analysis(trainer, X_train, X_test, feature_names,
                         top_n=20, save_dir=XAI_DIR):
    """
    Run SHAP analysis on ALL tree-based models and aggregate results
    in a robust way that handles different SHAP output shapes.

    Strategy:
        - For each model, compute mean(|SHAP|) per feature over samples.
        - Then average these per-feature vectors across models.
    """
    try:
        import shap
    except ImportError:
        print("\nSHAP not installed. Run: pip install shap --break-system-packages")
        return None

    print("\n" + "=" * 80)
    print("GLOBAL SHAP ANALYSIS ACROSS ALL TREE-BASED MODELS")
    print("=" * 80)

    per_model_feature_importance = []  # list of (n_features,) arrays
    raw_shap_values = []               # optional: store raw per-model SHAP arrays
    model_names = []

    for model_key, (model, preprocessors) in trainer.trained_models.items():
        # Only run on tree-based models (those that had feature_importances_)
        if not hasattr(model, 'feature_importances_'):
            print(f"Skipping {model_key}: no feature_importances_ (likely non-tree model)")
            continue

        print(f"\n→ Computing SHAP for {model_key}...")

        # Preprocess X_test with that model's preprocessors
        X_test_proc = trainer._apply_preprocessing(X_test, preprocessors, fit=False)

        # Make sure we have a numpy array
        X_test_np = X_test_proc.values if hasattr(X_test_proc, "values") else X_test_proc

        # Build explainer & compute SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_np)

        # Normalize SHAP output to a 2D array of shape (n_samples, n_features)
        # SHAP can return:
        #   - list of arrays (one per class)
        #   - a 3D array (n_classes, n_samples, n_features)
        #   - a 2D array (n_samples, n_features)
        sv = shap_values

        # Case 1: list-like (e.g., [class0, class1])
        if isinstance(sv, list):
            # Binary: use positive class (index 1) if available, else mean across classes
            if len(sv) == 2:
                sv = sv[1]
            else:
                # Average across classes: result (n_samples, n_features)
                sv = np.mean(np.stack(sv, axis=0), axis=0)

        # Convert to numpy array
        sv = np.array(sv)

        # Case 2: 3D array (n_classes, n_samples, n_features)
        if sv.ndim == 3:
            if sv.shape[0] == 2:
                # Binary: take positive class
                sv = sv[1]
            else:
                # General: average across classes
                sv = sv.mean(axis=0)  # → (n_samples, n_features)

        # At this point we expect sv to be 2D: (n_samples, n_features)
        if sv.ndim != 2:
            print(f"  ⚠️ Unexpected SHAP shape {sv.shape} for {model_key}, skipping this model.")
            continue

        if sv.shape[1] != len(feature_names):
            print(f"  ⚠️ Feature count mismatch for {model_key}: "
                  f"SHAP features={sv.shape[1]}, feature_names={len(feature_names)}. "
                  f"Skipping this model.")
            continue

        # Store raw SHAP values (optional, for later inspection)
        raw_shap_values.append(sv)
        model_names.append(model_key)

        # Compute mean |SHAP| over samples → per-feature importance
        mean_abs_by_feature = np.abs(sv).mean(axis=0)  # shape (n_features,)
        per_model_feature_importance.append(mean_abs_by_feature)

        print(f"  ✓ SHAP values shape (normalized): {sv.shape}")
        print(f"  ✓ Per-feature mean |SHAP| computed for {model_key}")

    if not per_model_feature_importance:
        print("\nNo valid tree-based models found for SHAP!")
        return None

    print(f"\n{'=' * 80}")
    print(f"AGGREGATING SHAP VALUES FROM {len(per_model_feature_importance)} MODELS")
    print('=' * 80)

    # Stack into (n_models, n_features) and average across models
    per_model_feature_importance = np.vstack(per_model_feature_importance)
    global_importance = per_model_feature_importance.mean(axis=0)  # (n_features,)

    # Build summary DataFrame
    shap_summary = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': global_importance
    }).sort_values('mean_abs_shap', ascending=False)

    print(f"\nTop {min(top_n, len(shap_summary))} Features by SHAP:")
    print(shap_summary.head(top_n).to_string(index=False))

    # Save CSV
    csv_path = os.path.join(save_dir, 'global_shap_importance.csv')
    shap_summary.to_csv(csv_path, index=False)
    print(f"\n✓ Saved SHAP summary to: {csv_path}")

    # Bar plot of top features
    plt.figure(figsize=(12, 8))
    top_features = shap_summary.head(top_n)
    plt.barh(range(len(top_features)), top_features['mean_abs_shap'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Mean |SHAP| (Aggregated Across All Models)', fontsize=12)
    plt.title(f'Global SHAP Feature Importance (Top {top_n})',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'global_shap_bar.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved SHAP bar plot: {save_path}")
    plt.close()

    # Return:
    #   - shap_summary: DataFrame with global importance
    #   - raw_shap_values: list of per-model SHAP arrays (possibly different shapes)
    #   - model_names: corresponding model keys
    return shap_summary, raw_shap_values, model_names



def compare_importance_methods(feature_imp_agg, shap_summary, top_n=15, save_dir=XAI_DIR):
    """
    Compare built-in feature importance vs SHAP importance
    """
    print("\n" + "=" * 80)
    print("COMPARING FEATURE IMPORTANCE METHODS")
    print("=" * 80)

    # Merge the two summaries
    merged = feature_imp_agg.reset_index().merge(
        shap_summary[['feature', 'mean_abs_shap']],
        on='feature',
        how='outer'
    )

    # Normalize both to 0-1 scale for comparison
    merged['norm_builtin'] = merged['mean_importance'] / merged['mean_importance'].max()
    merged['norm_shap'] = merged['mean_abs_shap'] / merged['mean_abs_shap'].max()

    # Get top features by either method
    top_by_builtin = set(merged.nlargest(top_n, 'norm_builtin')['feature'])
    top_by_shap = set(merged.nlargest(top_n, 'norm_shap')['feature'])
    top_features = list(top_by_builtin.union(top_by_shap))

    # Create comparison plot
    df_plot = merged[merged['feature'].isin(top_features)].copy()
    df_plot = df_plot.sort_values('norm_builtin', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 10))
    x = np.arange(len(df_plot))
    width = 0.35

    ax.barh(x - width / 2, df_plot['norm_builtin'], width, label='Built-in Importance', color='steelblue', alpha=0.8)
    ax.barh(x + width / 2, df_plot['norm_shap'], width, label='SHAP Importance', color='coral', alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(df_plot['feature'])
    ax.set_xlabel('Normalized Importance', fontsize=12)
    ax.set_title('Feature Importance: Built-in vs SHAP', fontsize=14, fontweight='bold')
    ax.legend()
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'importance_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {save_path}")
    plt.close()

    # Correlation between methods
    corr = merged[['norm_builtin', 'norm_shap']].corr().iloc[0, 1]
    print(f"\nCorrelation between methods: {corr:.3f}")

    # Agreement analysis
    agreement = len(top_by_builtin.intersection(top_by_shap))
    print(f"Features in top {top_n} for both methods: {agreement}/{top_n}")

    return merged


# ============================================================================
# MODEL PERFORMANCE CORRELATION WITH FEATURES
# ============================================================================

def analyze_feature_model_correlation(trainer, df_importance, save_dir=XAI_DIR):
    """
    Analyze if certain features correlate with better model performance
    """
    print("\n" + "=" * 80)
    print("ANALYZING FEATURE-PERFORMANCE CORRELATION")
    print("=" * 80)

    # Get model performance
    results_df = trainer.get_results()

    # Merge with feature importance
    merged = df_importance.merge(
        results_df[['model', 'test_f1_macro', 'cv_score']],
        on='model',
        how='left'
    )

    # For each feature, correlate its importance with model performance
    correlations = []
    for feature in merged['feature'].unique():
        feature_data = merged[merged['feature'] == feature]

        if len(feature_data) > 1:
            corr_test = feature_data['importance'].corr(feature_data['test_f1_macro'])
            corr_cv = feature_data['importance'].corr(feature_data['cv_score'])

            correlations.append({
                'feature': feature,
                'corr_with_test_f1': corr_test,
                'corr_with_cv_score': corr_cv,
                'mean_importance': feature_data['importance'].mean()
            })

    corr_df = pd.DataFrame(correlations).sort_values('corr_with_test_f1', ascending=False)

    print("\nFeatures Most Correlated with Performance:")
    print(corr_df.head(10).to_string(index=False))

    # Save
    csv_path = os.path.join(save_dir, 'feature_performance_correlation.csv')
    corr_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved correlations to: {csv_path}")

    return corr_df


# ============================================================================
# MASTER FUNCTION: EXPLAIN ALL MODELS
# ============================================================================

def explain_all_models(trainer, X_train, X_test, y_test, feature_names,
                       include_shap=True, top_n=20, save_dir=XAI_DIR):
    """
    COMPREHENSIVE XAI: Analyze ALL models globally

    This is your main function - it does everything!

    Args:
        trainer: ModelTrainer instance after fit_all() and evaluate()
        X_train: Training features (original, not preprocessed)
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        include_shap: Whether to run SHAP (slower but more accurate)
        top_n: Number of top features to display
        save_dir: Output directory

    Returns:
        Dictionary with all analysis results
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE XAI: GLOBAL ANALYSIS OF ALL MODELS")
    print("=" * 80)

    results = {}

    # 1. Extract feature importance from all tree models
    df_importance = extract_all_feature_importances(trainer, feature_names, save_dir)

    if df_importance is not None:
        # 2. Aggregate and analyze
        feature_imp_agg = aggregate_feature_importance(df_importance, top_n=top_n, save_dir=save_dir)
        results['feature_importance_agg'] = feature_imp_agg

        # 3. Visualize comparisons
        plot_comparative_feature_importance(df_importance, top_n=top_n, save_dir=save_dir)

        # 4. Identify consensus features
        consensus_features = identify_consensus_features(df_importance, top_n=15)
        results['consensus_features'] = consensus_features

        # 5. Correlate with model performance
        corr_df = analyze_feature_model_correlation(trainer, df_importance, save_dir)
        results['feature_performance_corr'] = corr_df

        # 6. SHAP analysis (optional, slower)
        if include_shap:
            shap_summary, shap_values, model_names = global_shap_analysis(
                trainer, X_train, X_test, feature_names, top_n=top_n, save_dir=save_dir
            )

            if shap_summary is not None:
                results['shap_summary'] = shap_summary
                results['shap_values'] = shap_values

                # Compare methods
                comparison = compare_importance_methods(
                    feature_imp_agg, shap_summary, top_n=15, save_dir=save_dir
                )
                results['importance_comparison'] = comparison

    print("\n" + "=" * 80)
    print("GLOBAL XAI COMPLETE!")
    print(f"All outputs saved to: {save_dir}/")
    print("=" * 80 + "\n")

    return results


# ============================================================================
# OPTIONAL: INDIVIDUAL SAMPLE EXPLANATION (LIME)
# ============================================================================

def explain_samples_with_lime(trainer, X_train, X_test, feature_names,
                              sample_indices=[0, 1, 2], save_dir=XAI_DIR):
    """
    Optional: Explain specific samples with LIME
    (Use this AFTER global analysis if you need to understand specific predictions)
    """
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        print("LIME not installed. Run: pip install lime --break-system-packages")
        return None

    print("\n" + "=" * 80)
    print(f"LIME EXPLANATIONS FOR {len(sample_indices)} SAMPLES")
    print("=" * 80)

    # Get best model
    best_model, preprocessors, best_stats = trainer.get_best_model(metric='test_f1_macro')
    model_name = best_stats['model'].replace('_', ' ').title()

    # Preprocess
    X_train_proc = trainer._apply_preprocessing(X_train, preprocessors, fit=False)
    X_test_proc = trainer._apply_preprocessing(X_test, preprocessors, fit=False)

    # Convert to numpy if needed
    X_train_np = X_train_proc.values if isinstance(X_train_proc, pd.DataFrame) else X_train_proc
    X_test_np = X_test_proc.values if isinstance(X_test_proc, pd.DataFrame) else X_test_proc

    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_np,
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
        mode='classification'
    )

    # Explain each sample
    for idx in sample_indices:
        if idx < len(X_test_np):
            print(f"\n→ Explaining sample {idx}...")

            explanation = explainer.explain_instance(
                X_test_np[idx],
                best_model.predict_proba,
                num_features=20
            )

            # Save plot
            fig = explanation.as_pyplot_figure()
            plt.title(f'LIME Explanation: {model_name} (Sample {idx})',
                      fontsize=14, fontweight='bold')
            plt.tight_layout()

            save_path = os.path.join(save_dir, f'lime_sample_{idx}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {save_path}")
            plt.close()

    print("\n" + "=" * 80)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\nEnhanced XAI Module Loaded Successfully!")
    print("\nMain function: explain_all_models()")
    print("\nTo install required packages:")
    print("  pip install shap lime matplotlib seaborn --break-system-packages")


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

    # Detect device from model
    # Try multiple ways to get the device
    try:
        # Method 1: Check if bert_model has a device attribute
        if hasattr(bert_model, 'device'):
            device = bert_model.device
        # Method 2: Get device from model parameters
        elif hasattr(bert_model, 'model'):
            device = next(bert_model.model.parameters()).device
        # Method 3: Default to cuda if available, else cpu
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"Warning: Could not detect device, defaulting to CPU. Error: {e}")
        device = torch.device('cpu')

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

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
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Enhanced XAI Module with BERT Support Loaded Successfully!")
    print("\nTo install required packages:")
    print("  pip install shap lime transformers --break-system-packages")


# ============================================================================
# GLOBAL EXPLAINABILITY FOR BERT AND FEATURE FUSION
# ============================================================================

def explain_bert_global(bert_model, X_test_text, y_test, n_samples=100,
                        num_features=30, model_name='BERT', save_dir=XAI_DIR):
    """
    GLOBAL explainability for BERT model by aggregating LIME across many samples

    Args:
        bert_model: Your BERTClassifier instance
        X_test_text: Test text samples
        y_test: Test labels
        n_samples: Number of samples to analyze for global patterns
        num_features: Number of top features to extract per sample
        model_name: Name for outputs
        save_dir: Directory to save outputs
    """
    try:
        import lime
        import lime.lime_text
    except ImportError:
        print("LIME not installed. Run: pip install lime --break-system-packages")
        return None

    print("\n" + "=" * 80)
    print(f"GLOBAL EXPLAINABILITY - {model_name}")
    print("=" * 80)
    print(f"Analyzing {n_samples} samples to find global patterns...")

    # Create explainer
    explainer = lime.lime_text.LimeTextExplainer(class_names=['Class 0', 'Class 1'])

    # Storage for word importances
    word_importance_class0 = {}  # words important for predicting class 0
    word_importance_class1 = {}  # words important for predicting class 1

    # Sample indices - get diverse samples from both classes
    class_0_indices = [i for i, label in enumerate(y_test) if label == 0]
    class_1_indices = [i for i, label in enumerate(y_test) if label == 1]

    n_per_class = min(n_samples // 2, len(class_0_indices), len(class_1_indices))
    import random
    random.seed(42)
    sample_indices = (random.sample(class_0_indices, n_per_class) +
                      random.sample(class_1_indices, n_per_class))

    print(f"Analyzing {len(sample_indices)} samples ({n_per_class} per class)...")

    # Analyze each sample
    for idx in tqdm(sample_indices, desc="Extracting LIME explanations"):
        text = X_test_text.iloc[idx] if isinstance(X_test_text, pd.Series) else X_test_text[idx]

        try:
            explanation = explainer.explain_instance(
                text,
                bert_model.predict_proba,
                num_features=num_features,
                num_samples=500  # LIME sampling
            )

            # Get feature weights for both classes
            for word, weight in explanation.as_list():
                # Positive weight means important for class 1
                if weight > 0:
                    word_importance_class1[word] = word_importance_class1.get(word, []) + [weight]
                else:
                    word_importance_class0[word] = word_importance_class0.get(word, []) + [abs(weight)]

        except Exception as e:
            print(f"Warning: Failed on sample {idx}: {e}")
            continue

    # Aggregate results
    print("\nAggregating results across all samples...")

    def aggregate_word_importance(word_dict):
        """Calculate mean, frequency, and total importance for each word"""
        results = []
        for word, weights in word_dict.items():
            results.append({
                'word': word,
                'mean_importance': np.mean(weights),
                'frequency': len(weights),
                'total_importance': np.sum(weights),
                'std_importance': np.std(weights)
            })
        return pd.DataFrame(results).sort_values('total_importance', ascending=False)

    df_class0 = aggregate_word_importance(word_importance_class0)
    df_class1 = aggregate_word_importance(word_importance_class1)

    # Save results
    print("\nSaving global importance results...")
    csv_path_0 = os.path.join(save_dir, f'{model_name}_global_words_class0.csv')
    csv_path_1 = os.path.join(save_dir, f'{model_name}_global_words_class1.csv')
    df_class0.to_csv(csv_path_0, index=False)
    df_class1.to_csv(csv_path_1, index=False)
    print(f"✓ Saved: {csv_path_0}")
    print(f"✓ Saved: {csv_path_1}")

    # Print top features
    print("\n" + "=" * 80)
    print(f"TOP 20 WORDS PREDICTING CLASS 0 (across {len(sample_indices)} samples)")
    print("=" * 80)
    print(df_class0.head(20).to_string())

    print("\n" + "=" * 80)
    print(f"TOP 20 WORDS PREDICTING CLASS 1 (across {len(sample_indices)} samples)")
    print("=" * 80)
    print(df_class1.head(20).to_string())

    # Create visualizations
    print("\nCreating global visualization...")

    # Top words bar plot for each class
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Class 0
    top_20_class0 = df_class0.head(20)
    axes[0].barh(range(len(top_20_class0)), top_20_class0['total_importance'].values, color='steelblue')
    axes[0].set_yticks(range(len(top_20_class0)))
    axes[0].set_yticklabels(top_20_class0['word'].values)
    axes[0].set_xlabel('Total Importance (Aggregated)', fontsize=12)
    axes[0].set_title(f'Top 20 Words for Class 0\n({model_name})', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # Class 1
    top_20_class1 = df_class1.head(20)
    axes[1].barh(range(len(top_20_class1)), top_20_class1['total_importance'].values, color='coral')
    axes[1].set_yticks(range(len(top_20_class1)))
    axes[1].set_yticklabels(top_20_class1['word'].values)
    axes[1].set_xlabel('Total Importance (Aggregated)', fontsize=12)
    axes[1].set_title(f'Top 20 Words for Class 1\n({model_name})', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_global_word_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {save_path}")
    plt.close()

    # Create frequency vs importance scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(df_class0['frequency'], df_class0['mean_importance'], alpha=0.5, color='steelblue')
    axes[0].set_xlabel('Frequency (# samples)', fontsize=12)
    axes[0].set_ylabel('Mean Importance', fontsize=12)
    axes[0].set_title(f'Word Frequency vs Importance - Class 0\n({model_name})', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Annotate top words
    for _, row in df_class0.head(10).iterrows():
        axes[0].annotate(row['word'], (row['frequency'], row['mean_importance']),
                         fontsize=8, alpha=0.7)

    axes[1].scatter(df_class1['frequency'], df_class1['mean_importance'], alpha=0.5, color='coral')
    axes[1].set_xlabel('Frequency (# samples)', fontsize=12)
    axes[1].set_ylabel('Mean Importance', fontsize=12)
    axes[1].set_title(f'Word Frequency vs Importance - Class 1\n({model_name})', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)

    # Annotate top words
    for _, row in df_class1.head(10).iterrows():
        axes[1].annotate(row['word'], (row['frequency'], row['mean_importance']),
                         fontsize=8, alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_global_frequency_scatter.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved scatter plot: {save_path}")
    plt.close()

    print("\n" + "=" * 80)
    print("GLOBAL EXPLAINABILITY COMPLETE!")
    print("=" * 80 + "\n")

    return df_class0, df_class1


def explain_feature_fusion_global(fusion_model, X_text_test, X_features_test, y_test,
                                  feature_names, n_samples=100, model_name='FeatureFusion',
                                  save_dir=XAI_DIR):
    """
    GLOBAL explainability for Feature Fusion model
    Analyzes both text contributions (via LIME) and manual feature contributions

    Args:
        fusion_model: Your BERTFeatureFusionClassifier instance
        X_text_test: Test text samples
        X_features_test: Test manual features
        y_test: Test labels
        feature_names: Names of manual features
        n_samples: Number of samples to analyze
        model_name: Name for outputs
        save_dir: Directory to save outputs
    """
    try:
        import lime
        import lime.lime_text
        import lime.lime_tabular
    except ImportError:
        print("LIME not installed. Run: pip install lime --break-system-packages")
        return None

    print("\n" + "=" * 80)
    print(f"GLOBAL EXPLAINABILITY - {model_name}")
    print("=" * 80)
    print(f"Analyzing BOTH text and manual features across {n_samples} samples...")

    # ========================================================================
    # PART 1: Analyze TEXT contributions
    # ========================================================================
    print("\n--- Analyzing Text Contributions ---")

    # Create wrapper for text-only predictions
    class TextOnlyWrapper:
        def __init__(self, fusion_model, X_features_test):
            self.fusion_model = fusion_model
            self.X_features_test = X_features_test

        def predict_proba(self, texts):
            """Predict using text + average features"""
            # Use mean features for all samples (to isolate text effect)
            mean_features = self.X_features_test.mean(axis=0).values
            n_samples = len(texts) if isinstance(texts, list) else 1
            features_batch = np.tile(mean_features, (n_samples, 1))

            return self.fusion_model.predict_proba(texts, features_batch)

    text_wrapper = TextOnlyWrapper(fusion_model, X_features_test)
    text_explainer = lime.lime_text.LimeTextExplainer(class_names=['Class 0', 'Class 1'])

    # Sample indices
    class_0_indices = [i for i, label in enumerate(y_test) if label == 0]
    class_1_indices = [i for i, label in enumerate(y_test) if label == 1]
    n_per_class = min(n_samples // 2, len(class_0_indices), len(class_1_indices))

    import random
    random.seed(42)
    sample_indices = (random.sample(class_0_indices, n_per_class) +
                      random.sample(class_1_indices, n_per_class))

    word_importance_class0 = {}
    word_importance_class1 = {}

    print(f"Analyzing text in {len(sample_indices)} samples...")
    for idx in tqdm(sample_indices, desc="Extracting text importance"):
        text = X_text_test.iloc[idx] if isinstance(X_text_test, pd.Series) else X_text_test[idx]

        try:
            explanation = text_explainer.explain_instance(
                text,
                text_wrapper.predict_proba,
                num_features=30,
                num_samples=500
            )

            for word, weight in explanation.as_list():
                if weight > 0:
                    word_importance_class1[word] = word_importance_class1.get(word, []) + [weight]
                else:
                    word_importance_class0[word] = word_importance_class0.get(word, []) + [abs(weight)]
        except Exception as e:
            print(f"Warning: Failed on sample {idx}: {e}")
            continue

    # Aggregate text results
    def aggregate_importance(word_dict):
        results = []
        for word, weights in word_dict.items():
            results.append({
                'word': word,
                'mean_importance': np.mean(weights),
                'frequency': len(weights),
                'total_importance': np.sum(weights)
            })
        return pd.DataFrame(results).sort_values('total_importance', ascending=False)

    df_text_class0 = aggregate_importance(word_importance_class0)
    df_text_class1 = aggregate_importance(word_importance_class1)

    # ========================================================================
    # PART 2: Analyze MANUAL FEATURE contributions
    # ========================================================================
    print("\n--- Analyzing Manual Feature Contributions ---")

    # Create wrapper for features-only predictions
    class FeaturesOnlyWrapper:
        def __init__(self, fusion_model, X_text_test):
            self.fusion_model = fusion_model
            # Create a generic text template
            self.generic_text = "This is a generic text sample."

        def predict_proba(self, features):
            """Predict using generic text + actual features"""
            n_samples = features.shape[0]
            texts = [self.generic_text] * n_samples
            return self.fusion_model.predict_proba(texts, features)

    features_wrapper = FeaturesOnlyWrapper(fusion_model, X_text_test)

    # Convert features to numpy
    X_features_np = X_features_test.values if isinstance(X_features_test, pd.DataFrame) else X_features_test

    # Create LIME tabular explainer for features
    features_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_features_np,
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
        mode='classification'
    )

    # PASTE THIS INTO YOUR xai_explanations.py
    # Replace lines 1132-1167 with this fixed version:

    feature_importance_class0 = {feat: [] for feat in feature_names}
    feature_importance_class1 = {feat: [] for feat in feature_names}

    print(f"Analyzing manual features in {len(sample_indices)} samples...")
    for idx in tqdm(sample_indices, desc="Extracting feature importance"):
        try:
            explanation = features_explainer.explain_instance(
                X_features_np[idx],
                features_wrapper.predict_proba,
                num_features=len(feature_names)
            )

            # LIME returns discretized names like 'fatalities > 0.00'
            # We need to extract the base feature name
            for feat, weight in explanation.as_list():
                # Extract base feature name by removing conditions
                base_feat = feat.split('<=')[0].split('>')[0].split('<')[0].split('=')[0].strip()

                # Only add if base feature is in our feature names
                if base_feat in feature_importance_class0:
                    if weight > 0:
                        feature_importance_class1[base_feat].append(weight)
                    else:
                        feature_importance_class0[base_feat].append(abs(weight))
                else:
                    # Skip features we don't recognize
                    pass
        except Exception as e:
            print(f"Warning: Failed on sample {idx}: {e}")
            continue

    # Aggregate feature results
    def aggregate_feature_importance(feat_dict):
        results = []
        for feat, weights in feat_dict.items():
            if weights:  # Only if we have data
                results.append({
                    'feature': feat,
                    'mean_importance': np.mean(weights),
                    'frequency': len(weights),
                    'total_importance': np.sum(weights),
                    'std_importance': np.std(weights)
                })

        # CRITICAL FIX: Handle empty results
        if not results:
            print(f"Warning: No feature importance data collected. Returning empty DataFrame.")
            return pd.DataFrame(
                columns=['feature', 'mean_importance', 'frequency', 'total_importance', 'std_importance'])

        return pd.DataFrame(results).sort_values('total_importance', ascending=False)

    df_feat_class0 = aggregate_feature_importance(feature_importance_class0)
    df_feat_class1 = aggregate_feature_importance(feature_importance_class1)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n--- Saving Results ---")

    # Save text results
    df_text_class0.to_csv(os.path.join(save_dir, f'{model_name}_text_class0.csv'), index=False)
    df_text_class1.to_csv(os.path.join(save_dir, f'{model_name}_text_class1.csv'), index=False)

    # Save feature results
    df_feat_class0.to_csv(os.path.join(save_dir, f'{model_name}_features_class0.csv'), index=False)
    df_feat_class1.to_csv(os.path.join(save_dir, f'{model_name}_features_class1.csv'), index=False)

    print(f"✓ Saved all results to {save_dir}/")

    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("TOP 15 WORDS FOR EACH CLASS")
    print("=" * 80)
    print("\nClass 0:")
    print(df_text_class0.head(15).to_string())
    print("\nClass 1:")
    print(df_text_class1.head(15).to_string())

    print("\n" + "=" * 80)
    print("TOP MANUAL FEATURES FOR EACH CLASS")
    print("=" * 80)
    print("\nClass 0:")
    print(df_feat_class0.head(15).to_string())
    print("\nClass 1:")
    print(df_feat_class1.head(15).to_string())

    # ========================================================================
    # CREATE VISUALIZATIONS
    # ========================================================================
    print("\n--- Creating Visualizations ---")

    # Combined visualization showing text AND features
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Top left: Text for Class 0
    top_15_text_0 = df_text_class0.head(15)
    axes[0, 0].barh(range(len(top_15_text_0)), top_15_text_0['total_importance'].values, color='steelblue')
    axes[0, 0].set_yticks(range(len(top_15_text_0)))
    axes[0, 0].set_yticklabels(top_15_text_0['word'].values, fontsize=10)
    axes[0, 0].set_xlabel('Total Importance', fontsize=11)
    axes[0, 0].set_title('Top Words Predicting Class 0', fontsize=13, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)

    # Top right: Text for Class 1
    top_15_text_1 = df_text_class1.head(15)
    axes[0, 1].barh(range(len(top_15_text_1)), top_15_text_1['total_importance'].values, color='coral')
    axes[0, 1].set_yticks(range(len(top_15_text_1)))
    axes[0, 1].set_yticklabels(top_15_text_1['word'].values, fontsize=10)
    axes[0, 1].set_xlabel('Total Importance', fontsize=11)
    axes[0, 1].set_title('Top Words Predicting Class 1', fontsize=13, fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis='x', alpha=0.3)

    # Bottom left: Features for Class 0
    top_15_feat_0 = df_feat_class0.head(15)
    axes[1, 0].barh(range(len(top_15_feat_0)), top_15_feat_0['total_importance'].values, color='mediumseagreen')
    axes[1, 0].set_yticks(range(len(top_15_feat_0)))
    axes[1, 0].set_yticklabels(top_15_feat_0['feature'].values, fontsize=10)
    axes[1, 0].set_xlabel('Total Importance', fontsize=11)
    axes[1, 0].set_title('Top Manual Features Predicting Class 0', fontsize=13, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis='x', alpha=0.3)

    # Bottom right: Features for Class 1
    top_15_feat_1 = df_feat_class1.head(15)
    axes[1, 1].barh(range(len(top_15_feat_1)), top_15_feat_1['total_importance'].values, color='orange')
    axes[1, 1].set_yticks(range(len(top_15_feat_1)))
    axes[1, 1].set_yticklabels(top_15_feat_1['feature'].values, fontsize=10)
    axes[1, 1].set_xlabel('Total Importance', fontsize=11)
    axes[1, 1].set_title('Top Manual Features Predicting Class 1', fontsize=13, fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis='x', alpha=0.3)

    plt.suptitle(f'Global Feature Importance: {model_name}\n(Text + Manual Features)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{model_name}_global_combined.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined visualization: {save_path}")
    plt.close()

    print("\n" + "=" * 80)
    print("FEATURE FUSION GLOBAL EXPLAINABILITY COMPLETE!")
    print("=" * 80 + "\n")

    return {
        'text_class0': df_text_class0,
        'text_class1': df_text_class1,
        'features_class0': df_feat_class0,
        'features_class1': df_feat_class1
    }