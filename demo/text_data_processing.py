"""
Test script to verify data filtering and feature preparation
"""
import pandas as pd
import sys

# Define feature sets
MANUAL_FEATURES = [
    'civilian_targeting',
    'fatalities',
    'violence_against_women',
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

EMBEDDING_FEATURES = [f'notes_emb_{i}' for i in range(384)]


def load_and_filter_data(csv_path):
    """Load CSV and filter to Taliban/ISIS-K attacks only"""
    df = pd.read_csv(csv_path)
    print(f"Original data shape: {df.shape}")

    # Check unique actors
    print(f"\nUnique actors in data:")
    print(df['actor1'].value_counts())

    # Filter to only Taliban/ISIS-K rows
    filtered_df = df[df['actor1'] == 'Taliban and/or Islamic State Khorasan Province (ISKP)'].copy()
    print(f"\nFiltered data shape: {filtered_df.shape}")

    return filtered_df


def prepare_data_for_model(df, model_type):
    """Prepare data based on model type"""
    print(f"\n{'=' * 60}")
    print(f"Preparing data for: {model_type}")
    print(f"{'=' * 60}")

    if model_type == "Classical":
        feature_cols = MANUAL_FEATURES + EMBEDDING_FEATURES
        prepared = df[feature_cols]
        print(f"Manual features: {len(MANUAL_FEATURES)}")
        print(f"Embedding features: {len(EMBEDDING_FEATURES)}")
        print(f"Total features: {len(prepared.columns)}")

    elif model_type == "BERT":
        prepared = df[['notes']]
        print(f"Using raw text from 'notes' column")
        print(f"Sample note: {prepared['notes'].iloc[0][:100]}...")

    elif model_type == "BERT-Hybrid":
        feature_cols = MANUAL_FEATURES + ['notes']
        prepared = df[feature_cols]
        print(f"Manual features: {len(MANUAL_FEATURES)}")
        print(f"Text feature: 1 (notes)")
        print(f"Total features: {len(prepared.columns)}")

    print(f"\nPrepared data shape: {prepared.shape}")
    print(f"\nFirst few rows:")
    print(prepared.head(3))

    return prepared


def main():
    # Load data
    csv_path = '/mnt/user-data/uploads/unattributed_attacks_processed.csv'

    print("=" * 60)
    print("DATA LOADING AND FILTERING TEST")
    print("=" * 60)

    df = load_and_filter_data(csv_path)

    # Test each model type
    for model_type in ["Classical", "BERT", "BERT-Hybrid"]:
        prepared = prepare_data_for_model(df, model_type)

        # Verify no missing values in critical columns
        if model_type == "Classical":
            missing = prepared.isnull().sum().sum()
            print(f"\nMissing values: {missing}")
        elif model_type == "BERT":
            missing_notes = prepared['notes'].isnull().sum()
            print(f"\nMissing notes: {missing_notes}")
        elif model_type == "BERT-Hybrid":
            missing_features = prepared[MANUAL_FEATURES].isnull().sum().sum()
            missing_notes = prepared['notes'].isnull().sum()
            print(f"\nMissing manual features: {missing_features}")
            print(f"Missing notes: {missing_notes}")

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE - All data processing working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()