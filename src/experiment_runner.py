#!/usr/bin/env python
"""
Simple Runner Script for Feature Fusion Experiments
Run this file to execute experiments with different configurations
"""

import sys
import os

# Add current directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_experiments import run_experiments, analyze_existing_results
import torch
import numpy as np


def main():
    print("\n" + "=" * 100)
    print("FEATURE FUSION EXPERIMENT RUNNER")
    print("=" * 100)
    print("\nSelect an option:")
    print("1. Run Quick Test (3 experiments, ~30 minutes)")
    print("2. Run Full Experiments (15 experiments, ~3 hours)")
    print("3. Analyze Existing Results")
    print("4. Run Specific Experiments")
    print("5. Exit")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice == '1':
        print("\n" + "-" * 80)
        print("Running Quick Test Experiments...")
        print("This will test the 3 most promising approaches")
        print("-" * 80)

        # Set seeds
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        run_experiments(experiment_set='quick', save_models=False)

    elif choice == '2':
        print("\n" + "-" * 80)
        print("Running Full Experiment Suite...")
        print("This will test all 15 configurations")
        print("WARNING: This may take 2-3 hours")
        print("-" * 80)

        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            # Set seeds
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            run_experiments(experiment_set='full', save_models=True)
        else:
            print("Cancelled.")

    elif choice == '3':
        print("\n" + "-" * 80)
        print("Analyzing Existing Results...")
        print("-" * 80)
        analyze_existing_results()

    elif choice == '4':
        print("\n" + "-" * 80)
        print("Available experiments:")
        print("-" * 80)

        experiments = [
            "baseline_focal",
            "class_weights_only",
            "batch_balance_only",
            "batch_balance_mild_weights",
            "batch_balance_focal_moderate",
            "batch_balance_focal_aggressive",
            "early_stop_optimal",
            "lower_threshold",
            "strong_regularization",
            "freeze_more_bert",
            "conservative_best",
            "aggressive_best",
            "adaptive_threshold",
            "low_lr_batch_balance",
            "warmup_schedule"
        ]

        for i, exp_name in enumerate(experiments, 1):
            print(f"{i:2}. {exp_name}")

        exp_nums = input("\nEnter experiment numbers to run (comma-separated, e.g., 1,4,11): ").strip()

        if exp_nums:
            try:
                selected_indices = [int(x.strip()) - 1 for x in exp_nums.split(',')]
                selected_experiments = [experiments[i] for i in selected_indices
                                        if 0 <= i < len(experiments)]

                if selected_experiments:
                    print(f"\nSelected experiments: {', '.join(selected_experiments)}")

                    # Set seeds
                    np.random.seed(42)
                    torch.manual_seed(42)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(42)

                    run_experiments(experiment_set='custom',
                                    save_models=False,
                                    specific_experiments=selected_experiments)
                else:
                    print("No valid experiments selected.")
            except:
                print("Invalid input. Please enter numbers separated by commas.")

    elif choice == '5':
        print("Exiting...")
        sys.exit(0)

    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU available. Training will be slow.")
        confirm = input("Continue without GPU? (y/n): ").strip().lower()
        if confirm != 'y':
            sys.exit(0)

    main()

    # Ask if user wants to run another experiment
    print("\n" + "=" * 100)
    again = input("\nRun another experiment? (y/n): ").strip().lower()
    if again == 'y':
        main()
    else:
        print("\nExperiment runner complete. Check 'experiment_results/' for detailed results.")
