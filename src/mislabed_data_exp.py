"""
Function to add to your main.py for exporting mis-labeled data
"""
import pandas as pd
import numpy as np


def export_mislabeled_data(X_text_test, X_feat_test, y_test, y_pred, y_proba,
                           output_filename='mislabeled_samples.csv'):
    """
    Export mis-labeled samples to CSV for manual inspection

    Args:
        X_text_test: Test text samples (Series or list)
        X_feat_test: Test manual features (DataFrame or array)
        y_test: True labels (Series or array)
        y_pred: Predicted labels (array)
        y_proba: Prediction probabilities (array with shape [n_samples, 2])
        output_filename: Name of output CSV file

    Returns:
        DataFrame of mis-labeled samples
    """
    # Identify mis-labeled samples
    mislabeled_mask = (y_test.values if hasattr(y_test, 'values') else y_test) != y_pred
    mislabeled_indices = np.where(mislabeled_mask)[0]

    print(f"\n{'=' * 80}")
    print(f"EXPORTING MIS-LABELED SAMPLES")
    print(f"{'=' * 80}")
    print(f"Total test samples: {len(y_test)}")
    print(f"Mis-labeled samples: {len(mislabeled_indices)} ({len(mislabeled_indices) / len(y_test) * 100:.2f}%)")

    if len(mislabeled_indices) == 0:
        print("No mis-labeled samples found!")
        return None

    # Get the mis-labeled data
    if hasattr(X_text_test, 'iloc'):
        mislabeled_texts = X_text_test.iloc[mislabeled_indices].values
    else:
        mislabeled_texts = [X_text_test[i] for i in mislabeled_indices]

    if hasattr(X_feat_test, 'iloc'):
        mislabeled_features = X_feat_test.iloc[mislabeled_indices]
    else:
        mislabeled_features = X_feat_test[mislabeled_indices]

    if hasattr(y_test, 'iloc'):
        mislabeled_true_labels = y_test.iloc[mislabeled_indices].values
    else:
        mislabeled_true_labels = y_test[mislabeled_indices]

    mislabeled_pred_labels = y_pred[mislabeled_indices]
    mislabeled_probs = y_proba[mislabeled_indices]

    # Create DataFrame
    mislabeled_df = pd.DataFrame({
        'text': mislabeled_texts,
        'true_label': mislabeled_true_labels,
        'predicted_label': mislabeled_pred_labels,
        'prob_class_0': mislabeled_probs[:, 0],
        'prob_class_1': mislabeled_probs[:, 1],
        'prediction_confidence': np.max(mislabeled_probs, axis=1)
    })

    # Add manual features
    if isinstance(mislabeled_features, pd.DataFrame):
        feature_cols = mislabeled_features.columns
        for col in feature_cols:
            mislabeled_df[f'feature_{col}'] = mislabeled_features[col].values
    else:
        for i in range(mislabeled_features.shape[1]):
            mislabeled_df[f'feature_{i}'] = mislabeled_features[:, i]

    # Add error analysis columns
    mislabeled_df['error_type'] = mislabeled_df.apply(
        lambda row: 'False Positive' if row['predicted_label'] == 1 else 'False Negative',
        axis=1
    )

    # Sort by prediction confidence (least confident first - these are most uncertain)
    mislabeled_df = mislabeled_df.sort_values('prediction_confidence', ascending=True)

    # Add index for easier reference
    mislabeled_df.insert(0, 'sample_index', mislabeled_indices)

    # Summary statistics
    print("\nMis-labeled Sample Breakdown:")
    print(f"False Positives (predicted 1, actually 0): {(mislabeled_df['error_type'] == 'False Positive').sum()}")
    print(f"False Negatives (predicted 0, actually 1): {(mislabeled_df['error_type'] == 'False Negative').sum()}")
    print(f"\nAverage prediction confidence on errors: {mislabeled_df['prediction_confidence'].mean():.4f}")
    print(f"Lowest confidence error: {mislabeled_df['prediction_confidence'].min():.4f}")
    print(f"Highest confidence error: {mislabeled_df['prediction_confidence'].max():.4f}")

    # Save to CSV
    mislabeled_df.to_csv(output_filename, index=False)
    print(f"\nMis-labeled samples saved to '{output_filename}'")
    print(f"{'=' * 80}\n")

    return mislabeled_df


def export_mislabeled_by_class(X_text_test, X_feat_test, y_test, y_pred, y_proba,
                               output_prefix='mislabeled'):
    """
    Export mis-labeled samples separated by error type (FP and FN)

    Args:
        X_text_test: Test text samples
        X_feat_test: Test manual features
        y_test: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        output_prefix: Prefix for output files

    Returns:
        Tuple of (false_positives_df, false_negatives_df)
    """
    # Get all mislabeled data
    mislabeled_df = export_mislabeled_data(
        X_text_test, X_feat_test, y_test, y_pred, y_proba,
        output_filename=f'{output_prefix}_all.csv'
    )

    if mislabeled_df is None:
        return None, None

    # Separate by error type
    false_positives = mislabeled_df[mislabeled_df['error_type'] == 'False Positive']
    false_negatives = mislabeled_df[mislabeled_df['error_type'] == 'False Negative']

    # Save separate files
    if len(false_positives) > 0:
        false_positives.to_csv(f'{output_prefix}_false_positives.csv', index=False)
        print(f"False positives saved to '{output_prefix}_false_positives.csv'")

    if len(false_negatives) > 0:
        false_negatives.to_csv(f'{output_prefix}_false_negatives.csv', index=False)
        print(f"False negatives saved to '{output_prefix}_false_negatives.csv'")

    return false_positives, false_negatives