"""
Main Experiments Runner
Systematically test different feature fusion configurations to optimize minority class recall
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse
import json

# Import experiment configurations
from feature_fusion_experiments import (
    ExperimentConfig,
    get_experiment_configs,
    get_quick_test_configs,
    save_experiment_results,
    print_experiment_summary,
    compare_to_baseline,
    apply_config_to_model,
    find_optimal_threshold
)

# Import existing modules
from data_loader import load_data
from feature_eng import feature_creating, mask_group_names, mask_location_names
from feature_fusion import BERTFeatureFusionClassifier

warnings.filterwarnings('ignore')

# Manual features configuration (same as main.py)
FUSION_MANUAL_FEATURES = [
    # Base numeric features
    'civilian_targeting',
    'fatalities',
    'violence_against_women',

    # Attack patterns
    'coordinated_attack',
    'series_attack',

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


def prepare_data():
    """Prepare data for experiments (same as in main.py)"""

    print("\n" + "=" * 80)
    print("PREPARING DATA FOR EXPERIMENTS")
    print("=" * 80)

    # 1. Load data
    print("\nLoading data...")
    df = load_data()

    # 2. Feature engineering
    print("Creating features...")
    working_df, unattrib_df = feature_creating(
        df,
        use_embeddings=False,
        text_columns=None
    )

    # 3. Get text and apply masking
    X_text = working_df['notes'].fillna('')
    print("\nApplying semantic masking...")
    X_text = X_text.apply(mask_group_names)
    X_text = X_text.apply(mask_location_names)

    # 4. Select manual features
    available_features = [col for col in FUSION_MANUAL_FEATURES if col in working_df.columns]
    X_manual = working_df[available_features].copy()
    print(f"Selected {len(available_features)} manual features")

    y = working_df['target']

    # 5. Train-test split
    X_text_train, X_text_test, X_man_train, X_man_test, y_train, y_test = train_test_split(
        X_text, X_manual, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # 6. Scale manual features
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

    # 7. Create validation split
    X_text_tr, X_text_val, X_man_tr, X_man_val, y_tr, y_val = train_test_split(
        X_text_train, X_man_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    print(f"\nData splits:")
    print(f"  Train: {len(X_text_tr)} samples")
    print(f"  Val: {len(X_text_val)} samples")
    print(f"  Test: {len(X_text_test)} samples")

    # Print class distribution
    print(f"\nClass distribution:")
    print(f"  Train - Class 0: {sum(y_tr == 0)}, Class 1: {sum(y_tr == 1)}")
    print(f"  Val - Class 0: {sum(y_val == 0)}, Class 1: {sum(y_val == 1)}")
    print(f"  Test - Class 0: {sum(y_test == 0)}, Class 1: {sum(y_test == 1)}")

    return {
        'X_text_tr': X_text_tr,
        'X_text_val': X_text_val,
        'X_text_test': X_text_test,
        'X_man_tr': X_man_tr,
        'X_man_val': X_man_val,
        'X_man_test': X_man_test,
        'y_tr': y_tr,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': available_features
    }


def run_single_experiment(exp_config: ExperimentConfig, data: dict, save_model: bool = False):
    """Run a single experiment with given configuration"""

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {exp_config.name}")
    print(f"Description: {exp_config.description}")
    print("=" * 80)

    try:
        # Create model with experimental configuration
        model = apply_config_to_model(BERTFeatureFusionClassifier, exp_config.config)

        # Print key configuration
        print("\nKey Configuration:")
        for key in ['focal_loss', 'use_batch_balancing', 'use_class_weights',
                    'epochs', 'prediction_threshold', 'learning_rate']:
            if key in exp_config.config:
                print(f"  {key}: {exp_config.config[key]}")

        # Train the model
        print("\nTraining model...")
        model.fit(
            data['X_text_tr'], data['X_man_tr'], data['y_tr'],
            data['X_text_val'], data['X_man_val'], data['y_val']
        )

        # Handle adaptive threshold
        if exp_config.config.get('use_adaptive_threshold', False):
            print("\nOptimizing threshold for target recall...")
            target_recall = exp_config.config.get('target_recall', 0.75)

            # Get validation predictions
            _, _, val_proba = model.evaluate(
                data['X_text_val'], data['X_man_val'], data['y_val']
            )

            # Find optimal threshold
            threshold_results = find_optimal_threshold(
                data['y_val'], val_proba, target_recall
            )

            optimal_threshold = threshold_results['threshold']
            print(f"  Optimal threshold: {optimal_threshold:.3f}")
            print(f"  Expected recall: {threshold_results['recall']:.1%}")
            print(f"  Expected precision: {threshold_results['precision']:.1%}")

            # Update model threshold
            model.prediction_threshold = optimal_threshold
        else:
            # Use regular threshold optimization
            model.find_optimal_threshold(
                data['X_text_val'], data['X_man_val'], data['y_val']
            )

        # Evaluate on test set
        print("\nEvaluating on test set...")
        results, y_pred, y_proba = model.evaluate(
            data['X_text_test'], data['X_man_test'], data['y_test']
        )

        # Store results in experiment
        exp_config.results = results

        # Print results
        print("\nResults:")
        print(f"  Minority Recall: {results['minority_recall']:.1%}")
        print(f"  Minority Precision: {results['minority_precision']:.1%}")
        print(f"  Minority F1: {results['minority_f1']:.3f}")
        print(f"  Overall Accuracy: {results['test_accuracy']:.1%}")
        print(f"  Macro F1: {results['test_f1_macro']:.3f}")

        # Save model if requested
        if save_model:
            model_path = f"experiment_models/{exp_config.name}_model.pt"
            os.makedirs("experiment_models", exist_ok=True)
            torch.save(model.model.state_dict(), model_path)
            print(f"\nModel saved to {model_path}")

        # Save training stats
        training_stats = model.get_training_stats()
        stats_path = f"experiment_results/{exp_config.name}_training_stats.csv"
        os.makedirs("experiment_results", exist_ok=True)
        training_stats.to_csv(stats_path, index=False)

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

        return exp_config.results

    except Exception as e:
        print(f"\nERROR in experiment {exp_config.name}: {str(e)}")
        exp_config.results = {'error': str(e)}
        return None


def run_experiments(experiment_set: str = 'quick',
                    save_models: bool = False,
                    specific_experiments: list = None):
    """
    Run a set of experiments

    Args:
        experiment_set: 'quick' for quick tests, 'full' for all experiments,
                       'custom' for specific experiments
        save_models: Whether to save trained models
        specific_experiments: List of experiment names to run (for 'custom' mode)
    """

    print("\n" + "=" * 100)
    print("FEATURE FUSION EXPERIMENTS RUNNER")
    print("=" * 100)
    print(f"Experiment Set: {experiment_set}")
    print(f"Save Models: {save_models}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Prepare data once for all experiments
    data = prepare_data()

    # Get experiment configurations
    if experiment_set == 'quick':
        experiments = get_quick_test_configs()
        print(f"\nRunning {len(experiments)} quick test experiments")
    elif experiment_set == 'full':
        experiments = get_experiment_configs()
        print(f"\nRunning {len(experiments)} full experiments")
    elif experiment_set == 'custom' and specific_experiments:
        all_experiments = get_experiment_configs()
        experiments = [exp for exp in all_experiments
                       if exp.name in specific_experiments]
        print(f"\nRunning {len(experiments)} specific experiments: {specific_experiments}")
    else:
        print("Invalid experiment set specified")
        return

    # Run experiments
    successful_experiments = []
    failed_experiments = []

    for i, exp in enumerate(experiments, 1):
        print(f"\n{'=' * 100}")
        print(f"Running Experiment {i}/{len(experiments)}")

        result = run_single_experiment(exp, data, save_models)

        if result:
            successful_experiments.append(exp)
        else:
            failed_experiments.append(exp)

    # Print summary
    print("\n" + "=" * 100)
    print("EXPERIMENTS COMPLETE")
    print("=" * 100)
    print(f"Successful: {len(successful_experiments)}/{len(experiments)}")
    print(f"Failed: {len(failed_experiments)}/{len(experiments)}")

    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp.name}: {exp.results.get('error', 'Unknown error')}")

    # Save results
    if successful_experiments:
        save_experiment_results(successful_experiments)
        print_experiment_summary(successful_experiments)
        compare_to_baseline(successful_experiments, baseline_recall=0.672)

        # Find and highlight best configuration
        best_exp = max(successful_experiments,
                       key=lambda x: x.results.get('minority_recall', 0))

        print("\n" + "=" * 100)
        print("BEST CONFIGURATION FOR MINORITY RECALL")
        print("=" * 100)
        print(f"Experiment: {best_exp.name}")
        print(f"Description: {best_exp.description}")
        print(f"Minority Recall: {best_exp.results['minority_recall']:.1%}")
        print(f"Minority Precision: {best_exp.results['minority_precision']:.1%}")
        print(f"Minority F1: {best_exp.results['minority_f1']:.3f}")
        print("\nConfiguration:")
        for key, value in best_exp.config.items():
            if key in ['focal_loss', 'use_batch_balancing', 'use_class_weights',
                       'focal_alpha', 'focal_gamma', 'class_weight_ratio',
                       'prediction_threshold', 'epochs', 'learning_rate']:
                print(f"  {key}: {value}")

        # Save best configuration
        best_config_path = "experiment_results/best_configuration.json"
        with open(best_config_path, 'w') as f:
            json.dump({
                'name': best_exp.name,
                'description': best_exp.description,
                'config': best_exp.config,
                'results': best_exp.results
            }, f, indent=2)
        print(f"\nBest configuration saved to {best_config_path}")

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def analyze_existing_results(results_dir: str = "experiment_results"):
    """Analyze existing experiment results without re-running"""

    print("\n" + "=" * 100)
    print("ANALYZING EXISTING EXPERIMENT RESULTS")
    print("=" * 100)

    # Load experiment summary
    summary_path = os.path.join(results_dir, "experiment_summary.csv")
    if not os.path.exists(summary_path):
        print(f"No results found in {results_dir}")
        return

    summary_df = pd.read_csv(summary_path)

    print(f"\nFound {len(summary_df)} experiment results")
    print("\n" + "-" * 80)
    print("ALL EXPERIMENTS SUMMARY")
    print("-" * 80)
    print(summary_df.to_string(index=False))

    # Analyze by category
    print("\n" + "-" * 80)
    print("TOP 5 BY MINORITY RECALL")
    print("-" * 80)
    top_recall = summary_df.nlargest(5, 'minority_recall')
    print(top_recall[['name', 'minority_recall', 'minority_precision', 'minority_f1']].to_string(index=False))

    print("\n" + "-" * 80)
    print("TOP 5 BY MINORITY F1")
    print("-" * 80)
    top_f1 = summary_df.nlargest(5, 'minority_f1')
    print(top_f1[['name', 'minority_recall', 'minority_precision', 'minority_f1']].to_string(index=False))

    # Analyze trends
    print("\n" + "-" * 80)
    print("CONFIGURATION ANALYSIS")
    print("-" * 80)

    # Group experiments by key features
    batch_balanced = summary_df[summary_df['name'].str.contains('batch_balance')]
    focal_loss = summary_df[summary_df['name'].str.contains('focal')]
    class_weights = summary_df[summary_df['name'].str.contains('class_weight')]

    if not batch_balanced.empty:
        print(f"\nBatch Balanced Experiments:")
        print(f"  Avg Minority Recall: {batch_balanced['minority_recall'].mean():.1%}")
        print(f"  Avg Minority Precision: {batch_balanced['minority_precision'].mean():.1%}")

    if not focal_loss.empty:
        print(f"\nFocal Loss Experiments:")
        print(f"  Avg Minority Recall: {focal_loss['minority_recall'].mean():.1%}")
        print(f"  Avg Minority Precision: {focal_loss['minority_precision'].mean():.1%}")

    if not class_weights.empty:
        print(f"\nClass Weighted Experiments:")
        print(f"  Avg Minority Recall: {class_weights['minority_recall'].mean():.1%}")
        print(f"  Avg Minority Precision: {class_weights['minority_precision'].mean():.1%}")


def main():
    """Main function with command-line interface"""

    parser = argparse.ArgumentParser(description='Run Feature Fusion Experiments')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full', 'custom', 'analyze'],
                        help='Experiment mode to run')
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models')
    parser.add_argument('--experiments', nargs='+', type=str,
                        help='Specific experiments to run (for custom mode)')
    parser.add_argument('--results-dir', type=str, default='experiment_results',
                        help='Directory for results (for analyze mode)')

    args = parser.parse_args()

    if args.mode == 'analyze':
        analyze_existing_results(args.results_dir)
    else:
        run_experiments(
            experiment_set=args.mode,
            save_models=args.save_models,
            specific_experiments=args.experiments
        )


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    main()
