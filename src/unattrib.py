"""
Process Unattributed Attacks
============================
This script processes unattributed attacks (no attribution) with the same
feature engineering pipeline applied to the training data.

Usage:
    python process_unattributed.py
"""

import pandas as pd
import sys
import os

# Import the required functions
from data_loader import load_data
from feature_eng import create_text_embeddings


def process_unattributed_attacks(use_embeddings=True, text_columns=None):
    """
    Process unattributed attacks with identical feature engineering as training data.

    Args:
        use_embeddings: Whether to create text embeddings (default: True)
        text_columns: List of text columns to embed (default: ['notes'])

    Returns:
        unattrib_processed: Processed DataFrame of unattributed attacks with all features
    """
    print("\n" + "=" * 70)
    print("PROCESSING UNATTRIBUTED ATTACKS")
    print("=" * 70)

    # Load the raw data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} total events")

    # Define unattributed groups
    unattrib_groups = [
        'Unidentified Armed Group (Afghanistan)',
        'Taliban and/or Islamic State Khorasan Province (ISKP)'
    ]

    # Filter for unattributed attacks
    print(f"\nFiltering for unattributed attacks...")
    unattrib_df = df[df['actor1'].isin(unattrib_groups)].copy()
    unattrib_df = unattrib_df.reset_index(drop=True)
    print(f"Found {len(unattrib_df)} unattributed attacks")

    if len(unattrib_df) == 0:
        print("WARNING: No unattributed attacks found!")
        return unattrib_df

    # Define violence against women tags (same as training pipeline)
    violence_against_women_tags = [
        'women targeted: government officials',
        'women targeted: girls',
        'women targeted: girls; women targeted: relatives of targeted groups or persons',
        'local administrators',
        'women targeted: government officials; women targeted: relatives of targeted groups or persons',
        'women targeted: candidates for office',
        'women targeted: activists/human rights defenders/social leaders',
        'women targeted: relatives of targeted groups or persons',
        'local administrators; women targeted: politicians',
        'women targeted: activists/human rights defenders/social leaders; women targeted: government officials'
    ]

    # Create violence against women feature
    print("\nCreating violence_against_women feature...")
    unattrib_df['violence_against_women'] = (
        (unattrib_df['sub_event_type'] == 'Sexual violence') |
        ((unattrib_df['tags'].isin(violence_against_women_tags)) &
         (unattrib_df['tags'] != 'local administrators'))
    ).astype(int)
    print(f"  {unattrib_df['violence_against_women'].sum()} events flagged for violence against women")

    # Process civilian_targeting column
    print("\nProcessing civilian_targeting feature...")
    col = "civilian_targeting"
    if col in unattrib_df.columns:
        unattrib_df[col] = unattrib_df[col].notna().astype(int)
        print(f"  {unattrib_df[col].sum()} events flagged for civilian targeting")
    else:
        print(f"  WARNING: Column '{col}' not found in data")

    # -----ADD TEXT EMBEDDINGS-----
    if use_embeddings and text_columns is not None:
        print("\n" + "=" * 60)
        print("CREATING TEXT EMBEDDINGS")
        print("=" * 60)

        # Create embeddings for specified text columns
        unattrib_df = create_text_embeddings(unattrib_df, text_columns)

        print("=" * 60)
        print(f"Total features after embeddings: {unattrib_df.shape[1]}")
        print("=" * 60 + "\n")

    # -----ONE-HOT ENCODING (after all feature creation)-----
    print("\nApplying one-hot encoding...")
    encoded_cols = ['sub_event_type']
    unattrib_df = pd.get_dummies(unattrib_df, columns=encoded_cols, dtype=int)
    print(f"  One-hot encoded: {encoded_cols}")

    # -----ENSURE ALL EXPECTED COLUMNS ARE PRESENT-----
    # Define all possible sub_event_type categories (from train_test_split.py)
    expected_sub_event_types = [
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

    print("\nEnsuring all expected sub_event_type columns are present...")
    missing_cols = []
    for col in expected_sub_event_types:
        if col not in unattrib_df.columns:
            unattrib_df[col] = 0  # Add missing column with all zeros
            missing_cols.append(col)

    if missing_cols:
        print(f"  Added {len(missing_cols)} missing columns (all zeros):")
        for col in missing_cols:
            print(f"    - {col}")
    else:
        print("  All expected columns already present!")

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nFinal shape: {unattrib_df.shape}")
    print(f"Total samples: {len(unattrib_df)}")

    return unattrib_df


def main():
    """
    Main execution function
    """
    # Configuration (matches actual training pipeline)
    text_columns = ['notes']  # Only 'notes' is used for embeddings
    use_embeddings = True

    # Process the unattributed attacks
    unattrib_processed = process_unattributed_attacks(
        use_embeddings=use_embeddings,
        text_columns=text_columns
    )

    # Save to CSV
    output_path = 'unattributed_attacks_processed.csv'
    print(f"\nSaving processed data to: {output_path}")
    unattrib_processed.to_csv(output_path, index=False)
    print("Saved successfully!")

    # Display sample info
    print("\n" + "=" * 70)
    print("SAMPLE INFORMATION")
    print("=" * 70)
    print(f"\nColumns ({len(unattrib_processed.columns)}):")
    for col in sorted(unattrib_processed.columns):
        print(f"  - {col}")

    print(f"\nFirst few rows:")
    print(unattrib_processed.head())

    return unattrib_processed


if __name__ == "__main__":
    unattrib_df = main()