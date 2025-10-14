import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import numpy as np

# Create output directory for plots
SAVE_DIR = 'visualizations'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Created directory: {SAVE_DIR}")


def plot_model_performance(results_df, metric='cv_score', save_dir=SAVE_DIR):
    """
    Plot bar chart comparing model performance

    Args:
        results_df: DataFrame with results from trainer.get_results()
        metric: Metric to plot (default: 'cv_score')
        save_dir: Directory to save plots
    """
    # Determine if we're plotting CV or test metrics
    if metric not in results_df.columns:
        available_metrics = [col for col in results_df.columns
                             if any(x in col for x in ['f1', 'accuracy', 'precision', 'recall', 'cv_score'])]
        print(f"Metric '{metric}' not found. Available: {available_metrics}")
        return

    plt.figure(figsize=(14, 6))
    sns.barplot(data=results_df, x='model', y=metric, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Model Comparison on {metric}', fontsize=14, fontweight='bold')
    plt.ylabel(metric, fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'model_performance_{metric}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_confusion_matrix_best(trainer, X_test, y_test, save_dir=SAVE_DIR):
    """
    Plot confusion matrix for best model

    Args:
        trainer: ModelTrainer instance
        X_test: Test features
        y_test: Test labels
        save_dir: Directory to save plots
    """
    try:
        # Get best model (now returns 3 values)
        model, preprocessors, stats = trainer.get_best_model()
        model_name = stats['model']

        # Preprocess test data using the model's preprocessors
        X_test_proc = trainer._apply_preprocessing(X_test, preprocessors, fit=False)

        # Make predictions
        y_pred = model.predict(X_test_proc)

        # Create confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            cmap='Blues',
            ax=ax,
            colorbar=True
        )
        ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'confusion_matrix_best.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        import traceback
        traceback.print_exc()


def plot_roc_pr(trainer, X_test, y_test, save_dir=SAVE_DIR):
    """
    Plot ROC and Precision-Recall curves for best model

    Args:
        trainer: ModelTrainer instance
        X_test: Test features
        y_test: Test labels
        save_dir: Directory to save plots
    """
    try:
        # Get best model (now returns 3 values)
        model, preprocessors, stats = trainer.get_best_model()
        model_name = stats['model']

        # Preprocess test data
        X_test_proc = trainer._apply_preprocessing(X_test, preprocessors, fit=False)

        # Get probability scores
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_proc)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test_proc)
        else:
            print(f"{model_name} does not support probability estimates.")
            return

        # Calculate curves
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        pr_auc = auc(recall, precision)

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROC Curve
        axes[0].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[0].set_title(f'ROC Curve: {model_name}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('False Positive Rate', fontsize=11)
        axes[0].set_ylabel('True Positive Rate', fontsize=11)
        axes[0].legend(loc='lower right')
        axes[0].grid(alpha=0.3)

        # Precision-Recall Curve
        axes[1].plot(recall, precision, linewidth=2, label=f'AUC = {pr_auc:.3f}')
        axes[1].set_title(f'Precision-Recall Curve: {model_name}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Recall', fontsize=11)
        axes[1].set_ylabel('Precision', fontsize=11)
        axes[1].legend(loc='lower left')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'roc_pr_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    except Exception as e:
        print(f"Error creating ROC/PR curves: {e}")
        import traceback
        traceback.print_exc()


def plot_cv_vs_test_comparison(results_df, save_dir=SAVE_DIR):
    """
    Plot comparison of CV scores vs test scores for all models

    Args:
        results_df: DataFrame with results
        save_dir: Directory to save plots
    """
    if 'test_f1_macro' not in results_df.columns:
        print("Test scores not available. Run trainer.evaluate() first.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(x - width / 2, results_df['cv_score'], width, label='CV F1', alpha=0.8)
    ax.bar(x + width / 2, results_df['test_f1_macro'], width, label='Test F1', alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Cross-Validation vs Test Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cv_vs_test_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_all_metrics(results_df, save_dir=SAVE_DIR):
    """
    Create a comprehensive plot with all metrics

    Args:
        results_df: DataFrame with results
        save_dir: Directory to save plots
    """
    # Check which metrics are available
    test_metrics = ['test_f1_macro', 'test_accuracy', 'test_precision', 'test_recall']
    available_test_metrics = [m for m in test_metrics if m in results_df.columns]

    if not available_test_metrics:
        print("No test metrics found. Run trainer.evaluate() first.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics = [
        ('test_f1_macro', 'F1 Score (Test)'),
        ('test_accuracy', 'Accuracy (Test)'),
        ('test_precision', 'Precision (Test)'),
        ('test_recall', 'Recall (Test)')
    ]

    for idx, (metric, title) in enumerate(metrics):
        if metric in results_df.columns:
            sns.barplot(data=results_df, x='model', y=metric, ax=axes[idx], palette='viridis')
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Model', fontsize=10)
            axes[idx].set_ylabel(title.split('(')[0].strip(), fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)
            for label in axes[idx].get_xticklabels():
                label.set_ha('right')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_all_plots(trainer, X_test, y_test, results_df=None, save_dir=SAVE_DIR):
    """
    Convenience function to generate all plots at once

    Args:
        trainer: ModelTrainer instance
        X_test: Test features
        y_test: Test labels
        results_df: Results DataFrame (if None, will call trainer.get_results())
        save_dir: Directory to save plots
    """
    if results_df is None:
        results_df = trainer.get_results()

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate all plots
    plot_model_performance(results_df, metric='cv_score', save_dir=save_dir)

    if 'test_f1_macro' in results_df.columns:
        plot_model_performance(results_df, metric='test_f1_macro', save_dir=save_dir)
        plot_cv_vs_test_comparison(results_df, save_dir=save_dir)
        plot_all_metrics(results_df, save_dir=save_dir)

    plot_confusion_matrix_best(trainer, X_test, y_test, save_dir=save_dir)
    plot_roc_pr(trainer, X_test, y_test, save_dir=save_dir)

    print("=" * 60)
    print(f"All visualizations saved to: {save_dir}")
    print("=" * 60 + "\n")