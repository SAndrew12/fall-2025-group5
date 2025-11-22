"""
Feature Fusion Experiments Configuration
Systematic testing of different configurations for optimal minority class recall
"""

import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime
import os


class ExperimentConfig:
    """Configuration container for experiments"""

    def __init__(self, name: str, config: Dict[str, Any], description: str = ""):
        self.name = name
        self.config = config
        self.description = description
        self.results = {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            'name': self.name,
            'config': self.config,
            'description': self.description,
            'results': self.results,
            'timestamp': self.timestamp
        }


def get_base_config():
    """Get the base configuration that all experiments build upon"""
    return {
        'model_name': 'bert-base-uncased',
        'max_length': 320,
        'batch_size': 16,
        'gradient_accumulation_steps': 2,
        'learning_rate': 2e-5,
        'epochs': 12,
        'random_state': 42,
        'early_stopping_patience': 3,
        'dropout_rate': 0.3,
        'hidden_dim': 128,
        'freeze_bert_base': False,
        'unfreeze_last_n_layers': 12,
        'prediction_threshold': 0.5,
    }


def get_experiment_configs():
    """Define all experiment configurations to test"""

    experiments = []
    base = get_base_config()

    # ============================================================================
    # EXPERIMENT SET 1: Loss Functions and Class Balancing
    # ============================================================================

    # 1. Baseline - Current best config (for reference)
    exp1 = base.copy()
    exp1.update({
        'focal_loss': True,
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'use_batch_balancing': False,
        'use_class_weights': False,
    })
    experiments.append(ExperimentConfig(
        "baseline_focal",
        exp1,
        "Current configuration with focal loss (baseline)"
    ))

    # 2. Pure class weights (no focal, no batch balance)
    exp2 = base.copy()
    exp2.update({
        'focal_loss': False,
        'use_batch_balancing': False,
        'use_class_weights': True,
        'class_weight_ratio': 49.0,  # For 49:1 imbalance
    })
    experiments.append(ExperimentConfig(
        "class_weights_only",
        exp2,
        "Simple class-weighted cross-entropy loss"
    ))

    # 3. Batch balancing only (no focal, no class weights)
    exp3 = base.copy()
    exp3.update({
        'focal_loss': False,
        'use_batch_balancing': True,
        'use_class_weights': False,
    })
    experiments.append(ExperimentConfig(
        "batch_balance_only",
        exp3,
        "Batch balancing without additional loss weighting"
    ))

    # 4. Batch balancing + mild class weights
    exp4 = base.copy()
    exp4.update({
        'focal_loss': False,
        'use_batch_balancing': True,
        'use_class_weights': True,
        'class_weight_ratio': 3.0,  # Mild weight since batches are balanced
    })
    experiments.append(ExperimentConfig(
        "batch_balance_mild_weights",
        exp4,
        "Batch balancing with mild class weights"
    ))

    # 5. Batch balancing + moderate focal loss
    exp5 = base.copy()
    exp5.update({
        'focal_loss': True,
        'focal_alpha': 0.6,  # Moderate alpha with batch balancing
        'focal_gamma': 2.0,
        'use_batch_balancing': True,
        'use_class_weights': False,
    })
    experiments.append(ExperimentConfig(
        "batch_balance_focal_moderate",
        exp5,
        "Batch balancing with moderate focal loss"
    ))

    # 6. Batch balancing + aggressive focal loss
    exp6 = base.copy()
    exp6.update({
        'focal_loss': True,
        'focal_alpha': 0.7,  # More aggressive alpha
        'focal_gamma': 2.5,  # Higher gamma for harder examples
        'use_batch_balancing': True,
        'use_class_weights': False,
        'learning_rate': 1e-5,  # Lower LR for stability
    })
    experiments.append(ExperimentConfig(
        "batch_balance_focal_aggressive",
        exp6,
        "Batch balancing with aggressive focal loss"
    ))

    # ============================================================================
    # EXPERIMENT SET 2: Thresholds and Early Stopping
    # ============================================================================

    # 7. Optimal early stopping point
    exp7 = base.copy()
    exp7.update({
        'focal_loss': True,
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'use_batch_balancing': False,
        'use_class_weights': False,
        'epochs': 9,  # Stop at peak validation performance
        'early_stopping_patience': 2,
    })
    experiments.append(ExperimentConfig(
        "early_stop_optimal",
        exp7,
        "Stop at epoch 9 where validation peaked"
    ))

    # 8. Lower prediction threshold
    exp8 = base.copy()
    exp8.update({
        'focal_loss': True,
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'use_batch_balancing': False,
        'use_class_weights': False,
        'prediction_threshold': 0.3,  # Lower threshold for more recall
    })
    experiments.append(ExperimentConfig(
        "lower_threshold",
        exp8,
        "Lower prediction threshold for higher recall"
    ))

    # ============================================================================
    # EXPERIMENT SET 3: Regularization
    # ============================================================================

    # 9. Stronger regularization
    exp9 = base.copy()
    exp9.update({
        'focal_loss': True,
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'use_batch_balancing': False,
        'use_class_weights': False,
        'dropout_rate': 0.4,
        'weight_decay': 0.01,
        'epochs': 9,
    })
    experiments.append(ExperimentConfig(
        "strong_regularization",
        exp9,
        "Stronger dropout and weight decay"
    ))

    # 10. More frozen BERT layers
    exp10 = base.copy()
    exp10.update({
        'focal_loss': True,
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'use_batch_balancing': False,
        'use_class_weights': False,
        'freeze_bert_base': True,
        'unfreeze_last_n_layers': 4,  # Only fine-tune top 4 layers
        'epochs': 9,
    })
    experiments.append(ExperimentConfig(
        "freeze_more_bert",
        exp10,
        "Freeze more BERT layers to reduce overfitting"
    ))

    # ============================================================================
    # EXPERIMENT SET 4: Combined Best Practices
    # ============================================================================

    # 11. Conservative best - balanced approach
    exp11 = base.copy()
    exp11.update({
        'focal_loss': False,
        'use_batch_balancing': True,
        'use_class_weights': True,
        'class_weight_ratio': 2.0,
        'epochs': 9,
        'early_stopping_patience': 2,
        'dropout_rate': 0.35,
        'prediction_threshold': 0.4,
    })
    experiments.append(ExperimentConfig(
        "conservative_best",
        exp11,
        "Conservative combination of best practices"
    ))

    # 12. Aggressive best - maximum recall push
    exp12 = base.copy()
    exp12.update({
        'focal_loss': True,
        'focal_alpha': 0.65,
        'focal_gamma': 2.5,
        'use_batch_balancing': True,
        'use_class_weights': False,
        'epochs': 9,
        'early_stopping_patience': 2,
        'dropout_rate': 0.4,
        'prediction_threshold': 0.3,
        'learning_rate': 1e-5,
    })
    experiments.append(ExperimentConfig(
        "aggressive_best",
        exp12,
        "Aggressive combination for maximum recall"
    ))

    # 13. Adaptive threshold (will be set dynamically)
    exp13 = base.copy()
    exp13.update({
        'focal_loss': False,
        'use_batch_balancing': True,
        'use_class_weights': True,
        'class_weight_ratio': 3.0,
        'epochs': 9,
        'early_stopping_patience': 2,
        'dropout_rate': 0.35,
        'use_adaptive_threshold': True,  # Will optimize on validation
        'target_recall': 0.75,  # Target 75% recall
    })
    experiments.append(ExperimentConfig(
        "adaptive_threshold",
        exp13,
        "Dynamically optimize threshold for 75% recall"
    ))

    # ============================================================================
    # EXPERIMENT SET 5: Learning Rate and Optimization
    # ============================================================================

    # 14. Lower learning rate with batch balancing
    exp14 = base.copy()
    exp14.update({
        'focal_loss': False,
        'use_batch_balancing': True,
        'use_class_weights': True,
        'class_weight_ratio': 2.5,
        'learning_rate': 5e-6,  # Much lower LR
        'epochs': 15,  # More epochs due to slower learning
        'early_stopping_patience': 3,
    })
    experiments.append(ExperimentConfig(
        "low_lr_batch_balance",
        exp14,
        "Lower learning rate with batch balancing"
    ))

    # 15. Warm-up learning rate schedule
    exp15 = base.copy()
    exp15.update({
        'focal_loss': False,
        'use_batch_balancing': True,
        'use_class_weights': True,
        'class_weight_ratio': 3.0,
        'use_warmup': True,
        'warmup_steps': 500,
        'epochs': 10,
    })
    experiments.append(ExperimentConfig(
        "warmup_schedule",
        exp15,
        "Learning rate warm-up for stable training"
    ))

    return experiments


def get_quick_test_configs():
    """Get a smaller set of configs for quick testing"""
    experiments = []
    base = get_base_config()

    # Just test the most promising approaches
    configs = [
        {
            'name': 'quick_class_weights',
            'desc': 'Class weights only',
            'updates': {
                'focal_loss': False,
                'use_batch_balancing': False,
                'use_class_weights': True,
                'class_weight_ratio': 49.0,
                'epochs': 6,
            }
        },
        {
            'name': 'quick_batch_balance',
            'desc': 'Batch balance + mild weights',
            'updates': {
                'focal_loss': False,
                'use_batch_balancing': True,
                'use_class_weights': True,
                'class_weight_ratio': 3.0,
                'epochs': 6,
            }
        },
        {
            'name': 'quick_focal_moderate',
            'desc': 'Batch balance + moderate focal',
            'updates': {
                'focal_loss': True,
                'focal_alpha': 0.6,
                'focal_gamma': 2.0,
                'use_batch_balancing': True,
                'use_class_weights': False,
                'epochs': 6,
            }
        },
    ]

    for cfg in configs:
        exp = base.copy()
        exp.update(cfg['updates'])
        experiments.append(ExperimentConfig(cfg['name'], exp, cfg['desc']))

    return experiments


def create_loss_function(config: Dict[str, Any], y_train: np.ndarray = None):
    """Create the appropriate loss function based on configuration"""

    if config.get('focal_loss', False):
        from feature_fusion import FocalLoss
        return FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        )

    elif config.get('use_class_weights', False):
        # Calculate class weights
        if y_train is not None:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            # Override with manual ratio if specified
            if 'class_weight_ratio' in config:
                class_weights = np.array([1.0, config['class_weight_ratio']])
        else:
            # Use manual ratio
            class_weights = np.array([1.0, config.get('class_weight_ratio', 1.0)])

        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight_tensor)

    else:
        # Standard cross-entropy
        return nn.CrossEntropyLoss()


def find_optimal_threshold(y_true, y_proba, target_recall=0.75):
    """Find the optimal threshold for a target recall"""
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba[:, 1])

    # Find threshold for target recall
    valid_indices = np.where(recalls >= target_recall)[0]
    if len(valid_indices) > 0:
        idx = valid_indices[-1]
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
        precision_at_threshold = precisions[idx]
        recall_at_threshold = recalls[idx]
    else:
        # If target recall can't be achieved, use the threshold with highest recall
        idx = np.argmax(recalls)
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
        precision_at_threshold = precisions[idx]
        recall_at_threshold = recalls[idx]

    return {
        'threshold': optimal_threshold,
        'precision': precision_at_threshold,
        'recall': recall_at_threshold,
        'f1': 2 * (precision_at_threshold * recall_at_threshold) /
              (precision_at_threshold + recall_at_threshold) if (
                                                                            precision_at_threshold + recall_at_threshold) > 0 else 0
    }


def save_experiment_results(experiments: List[ExperimentConfig], output_dir: str = "experiment_results"):
    """Save all experiment results to disk"""

    os.makedirs(output_dir, exist_ok=True)

    # Save individual experiment details
    for exp in experiments:
        exp_file = os.path.join(output_dir, f"{exp.name}_results.json")
        with open(exp_file, 'w') as f:
            json.dump(exp.to_dict(), f, indent=2)

    # Create summary DataFrame
    summary_data = []
    for exp in experiments:
        row = {
            'name': exp.name,
            'description': exp.description,
            **exp.results
        }
        summary_data.append(row)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, "experiment_summary.csv"), index=False)

        # Sort by minority recall and save top performers
        if 'minority_recall' in summary_df.columns:
            top_performers = summary_df.nlargest(5, 'minority_recall')
            top_performers.to_csv(os.path.join(output_dir, "top_performers.csv"), index=False)

        print(f"\nResults saved to {output_dir}/")
        return summary_df

    return None


def print_experiment_summary(experiments: List[ExperimentConfig]):
    """Print a nice summary of all experiments"""

    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)

    # Create summary table
    summary = []
    for exp in experiments:
        if exp.results:
            summary.append({
                'Name': exp.name[:25],
                'Minority Recall': f"{exp.results.get('minority_recall', 0):.1%}",
                'Minority Precision': f"{exp.results.get('minority_precision', 0):.1%}",
                'Minority F1': f"{exp.results.get('minority_f1', 0):.3f}",
                'Accuracy': f"{exp.results.get('test_accuracy', 0):.1%}",
            })

    if summary:
        df = pd.DataFrame(summary)
        print(df.to_string(index=False))

        # Find best performer
        best_recall_exp = max(experiments,
                              key=lambda x: x.results.get('minority_recall', 0) if x.results else 0)

        print("\n" + "-" * 100)
        print(f"BEST MINORITY RECALL: {best_recall_exp.name}")
        print(f"Recall: {best_recall_exp.results.get('minority_recall', 0):.1%}")
        print(f"Configuration: {best_recall_exp.description}")
        print("-" * 100)


def compare_to_baseline(experiments: List[ExperimentConfig], baseline_recall: float = 0.672):
    """Compare all experiments to baseline performance"""

    print("\n" + "=" * 80)
    print("COMPARISON TO BASELINE")
    print(f"Baseline Minority Recall: {baseline_recall:.1%}")
    print("=" * 80)

    improvements = []
    for exp in experiments:
        if exp.results and 'minority_recall' in exp.results:
            recall = exp.results['minority_recall']
            improvement = recall - baseline_recall
            improvements.append({
                'Name': exp.name,
                'Recall': f"{recall:.1%}",
                'Improvement': f"{improvement:+.1%}",
                'Status': '✅' if improvement > 0 else '❌'
            })

    if improvements:
        df = pd.DataFrame(improvements)
        df = df.sort_values('Improvement', ascending=False)
        print(df.to_string(index=False))


# Utility function to update model configuration
def apply_config_to_model(model_class, config: Dict[str, Any]):
    """Apply experimental configuration to the model"""

    # Map our experimental config to model parameters
    model_config = {
        'model_name': config.get('model_name', 'bert-base-uncased'),
        'max_length': config.get('max_length', 320),
        'batch_size': config.get('batch_size', 16),
        'learning_rate': config.get('learning_rate', 2e-5),
        'epochs': config.get('epochs', 12),
        'random_state': config.get('random_state', 42),
        'early_stopping_patience': config.get('early_stopping_patience', 3),
        'focal_loss': config.get('focal_loss', False),
        'focal_alpha': config.get('focal_alpha', 0.25),
        'focal_gamma': config.get('focal_gamma', 2.0),
        'freeze_bert_base': config.get('freeze_bert_base', False),
        'unfreeze_last_n_layers': config.get('unfreeze_last_n_layers', 12),
        'dropout_rate': config.get('dropout_rate', 0.3),
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 2),
        'prediction_threshold': config.get('prediction_threshold', 0.5),
        'use_batch_balancing': config.get('use_batch_balancing', False),
        'hidden_dim': config.get('hidden_dim', 128),
    }

    # Handle special configurations
    if config.get('use_class_weights', False):
        model_config['use_class_weights'] = True
        model_config['class_weight_ratio'] = config.get('class_weight_ratio', 1.0)

    if config.get('weight_decay'):
        model_config['weight_decay'] = config['weight_decay']

    if config.get('use_warmup'):
        model_config['use_warmup'] = True
        model_config['warmup_steps'] = config.get('warmup_steps', 500)

    return model_class(**model_config)
