import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re

EMBEDDING_MODEL = None


def mask_group_names(text):
    """
    Remove all variants of Taliban and ISIS-K group names entirely.

    Args:
        text: Input text string

    Returns:
        Text with group names removed
    """
    if not isinstance(text, str):
        return text

    # Define all variants (case-insensitive)
    taliban_variants = [
        r'\bTaliban\b',
        r'\bTaleban\b',
        r'\bTaliban-e\b',
        r'\bT-Ban\b',
        r'\bTban\b',
        r'\bTTP\b',  # Tehrik-i-Taliban Pakistan
        r'\bTehrik-i-Taliban\b',
        r'\bTehreek-e-Taliban\b',
    ]

    isis_variants = [
        r'\bISIS-K\b',
        r'\bISIS-KP\b',
        r'\bISIS Khorasan\b',
        r'\bISIS-Khorasan\b',
        r'\bISIL-K\b',
        r'\bISIL Khorasan\b',
        r'\bIS-K\b',
        r'\bIS-KP\b',
        r'\bIslamic State Khorasan\b',
        r'\bIslamic State in Khorasan\b',
        r'\bIslamic State of Khorasan\b',
        r'\bDaesh\b',
        r'\bDa\'esh\b',
        r'\bISIS\b',
        r'\bISIL\b',
        r'\bIslamic State\b',
        r'\bIS\b'
    ]

    # Combine all variants
    all_variants = taliban_variants + isis_variants

    # Remove each variant completely
    masked_text = text
    for pattern in all_variants:
        masked_text = re.sub(pattern, '', masked_text, flags=re.IGNORECASE)

    # Clean up multiple spaces and extra whitespace
    masked_text = re.sub(r'\s+', ' ', masked_text).strip()

    return masked_text


def mask_location_names(text):
    """
    Remove all location names (provinces and districts) from text.

    Args:
        text: Input text string

    Returns:
        Text with location names removed
    """
    if not isinstance(text, str):
        return text

    # Provinces
    provinces = ['Wardak', 'Shewa', 'Farah', 'Helmand', 'Kandahar', 'Nimruz', 'Paktia', 'Paktika',
                 'Parwan', 'Logar', 'Kunar', 'Ghazni', 'Laghman', 'Herat', 'Kabul', 'Sar-e Pol',
                 'Nangarhar', 'Nangahar', 'Khost', 'Zabul', 'Badakhshan', 'Faryab', 'Kunduz', 'Badghis',
                 'Balkh', 'Kapisa', 'Baghlan', 'Samangan', 'Urozgan', 'Jowzjan', 'Daykundi',
                 'Nuristan', 'Takhar', 'Ghor', 'Bamyan', 'Panjshir', 'Khatlon']

    # Districts (this is a large list)
    districts = ['Sayyid Abad', 'Bala Buluk', 'Lashkargah', 'Nahr-i-Saraj', 'Maruf', 'Khashrod',
                 'Zurmat', 'Sar Rawza', 'Bagram', 'Khak-i-Safed', 'Baraki Barak', 'Sangin',
                 'Sar Kani', 'Waghaz', 'Charkh', 'Dawlat Shah', 'Kushk-i-Kuhna', 'Surubi',
                 'Herat', 'Sancharak', 'Shinwar', 'Ghazi Abad', 'Andar', 'Nawa-i-Barikzayi',
                 'Nad Ali', 'Mata Khan', 'Puli Alam', 'Khost', 'Shemel Zayi', 'Gelan', 'Maiwand',
                 'Ghazni', 'Bati Kot', 'Faiz Abad', 'Gardez', 'Almar', 'Dangam', 'Rodat', 'Nesh',
                 'Spin Boldak', 'Mohammad Agha', 'Farah', 'Khan Abad', 'Shindand', 'Pushtrud',
                 'Ab Kamari', 'Ghormach', 'Mazar-e-Sharif', 'Qarabagh', 'Adraskan', 'Sholgara',
                 'Arghandab', 'Nijrab', 'Mehterlam', 'Yosuf Khel', 'Achin', 'Kandahar', 'Kot',
                 'Muqur', 'Deh Yak', 'Shah Wali Kot', 'Tagab', 'Qaisar', 'Asad Abad', 'Mizan',
                 'Pachir Waagam', 'Jalalabad', 'Surkh Rud', 'Garm Ser', 'Qarghayee', 'Chak-e-Wardak',
                 'Tala Wa Barfak', 'Hazrat-e-Sultan', 'Omna', 'Shah Joi', 'Kunduz', 'Bakwa', 'Kajaki',
                 'Sabari', 'Gurbuz', 'Kabul', 'Tarang Wa Jaldak', 'Qala-e-Zal', 'Baharak', 'Jani Khel',
                 'Manduzay', 'Khugyani', 'Washer', 'Shinkai', 'Nawzad', 'Sayyid Karam', 'Maidan Shahr',
                 'Naw Bahar', 'Wazakhwah', 'Shigal', 'Sher Zad', 'Behsud', 'Bala Murghab', 'Chaparhar',
                 'Dand Patan', 'Kuzkunar', 'Alasai', 'Ziruk', 'Hesarak', 'Dehdadi', 'Tirinkot',
                 'Alishing', 'Koh Band', 'Baghlan-e-Jadeed', 'Kohistan', 'Tirzayee', 'Lalpoor',
                 'Sheberghan', 'Deh Bala', 'Dasht-e-Archi', 'Dawlat Abad', 'Pul-i-Khumri', 'Watapoor',
                 'Qush Tepa', 'Chora', 'Hazrati Imam Sahib', 'Shirin Tagab', 'Maimana', 'Gulistan',
                 'Pashtun Kot', 'Muhmand Dara', 'Gizab', 'Kama', 'Char Bolak', 'Nawa', 'Sozma Qala',
                 'Daimir Dad', 'Goshta', 'Mara wara', 'Pur Chaman', 'Khulm', 'Kohistanat', 'Aybak',
                 'Noorgal', 'Kamdesh', 'Giro', 'Qalandar', 'Yamgan', 'Shamul', 'Nadir Shah Kot',
                 'Daman', 'Qadis', 'Musa Khel', 'Qala-i-Now', 'Ahmadabad', 'Narang Wa Badil',
                 'Shahidhassas', 'Alingar', 'Gurziwan', 'Karrukh', 'Nari', 'Darzab', 'Urgoon',
                 'Khwaja Ghar', 'Chishti Sharif', 'Mahmood Raqi', 'Shorabak', 'Noor Gram', 'Ajristan',
                 'Duab', 'Jaji Maidan', 'Pashtun Zarghun', 'Darqad', 'Dahana-e-Ghuri', 'Khakrez',
                 'Lash-i-Juwayn', 'Atghar', 'Kejran', 'Dushi', 'Chahar Darah', 'Darah-e-Noor',
                 'Barmal', 'Zarghun Shahr', 'Charikar', 'Ghurband', 'Khas Urozgan', 'Shibkoh',
                 'Khwaja Sabz Posh', 'Fersi', 'Kushk', 'Shakar Dara', 'Dila', 'Jalrez', 'Nerkh',
                 'Jaji', 'Wardooj', 'Bargi Matal', 'Yangi Qala', 'Chapa Dara', 'Qalat', 'Obe',
                 'Gulran', 'Sayyad', 'Zhire', 'Waygal', 'Anar Dara', 'Chimtal', 'Daichopan', 'Ab Band',
                 'Kaldar', 'Dehraoud', 'Bilchiragh', 'Khwaja Bahawuddin', 'Sharan', 'Paroon',
                 'Shaykh Ali', 'Sawkai', 'Jaghatu', 'Ali Abad', 'Jawand', 'Kharwar', 'Shortepa',
                 'Andkhoy', 'Aqchah', 'Arghistan', 'Kohsan', 'Saghar', 'Char Burjak', 'Chighcheran',
                 'Dara-e-Pech', 'Jurm', 'Balkh', 'Spera', 'Sar-e-Pul', 'Faizabad', 'Baak', 'Guzera',
                 'Eshkamesh', 'Tulak', 'Jabulussaraj', 'Samkani', 'Laja Ahmad khel', 'Sayyid Khel',
                 'Ghoryan', 'Miyanishin', 'Khwaja Omari', 'Wuza Zadran', 'Khwaja Hejran', 'Koh-e-Safi',
                 'Musahi', 'Zebak', 'Saighan', 'Eshkashim', 'Jaghuri', 'Paghman', 'Shahrak', 'Kishm',
                 'Zendahjan', 'Ghorak', 'Burka', 'Duleena', 'Taywara', 'Yawan', 'Giyan', 'Nahreen',
                 'Bazarak', 'Kalakan', 'Mingajik', 'Nazyan', 'Darayim', 'Deh Sabz', 'Gomal', 'Khash',
                 'Khak-e-Jabar', 'Qurghan', 'Chahar Asyab', 'Azra', 'Wama', 'Zaranj', 'Yahya Khel',
                 'Arghanj Khwah', 'Kahmard', 'Argo', 'Tanay', 'Asl-i-chakhansur', 'Khoshi',
                 'Dashti Qala', 'Shinwari', 'Nili', 'Tashkan', 'Malistan', 'Hissa-e-awali Behsud',
                 'Shuhada', 'Raghistan', 'Pasaband', 'Dara Sof Balla', 'Gozargah-e-Noor', 'Nahri Shahi',
                 'Yaftal-e-Sufla', 'Khinjan', 'Khanaqa', 'Mardyan', 'Khamyab', 'Zanakhan', 'Char Kent',
                 'Hissa-e-Awali Kohistan', 'Enjil', 'Mir Bacha Kot', 'Dur Baba', 'Shebar',
                 'Khost Wa Firing', 'Turwo', 'Lal Wa Sar Jangal', 'Panjwayee', 'Nika', 'Wormamay',
                 'Qarqin', 'Farza', 'Bagrami', 'Estalef', 'Reg-i-Khan Nishin', 'Kishindeh',
                 'Khani Charbagh', 'Qaram Qul', 'Qala-i-kah', 'Musa Qala', 'Namak Ab',
                 'Khuram Wa Sarbagh', 'Kufab', 'Khwaja Dukoh', 'Taluqan', 'Zari', 'Chahab', 'Reg',
                 'Ruyi Du Ab', 'Char Sada', 'Nawur', 'Shwak', 'Baghran', 'Rashidan', 'Darah',
                 'Dangara', 'Salang', 'Markaz-e-Behsud', 'Sang-e-Takht', 'Gosfandi', 'Shahri Buzurg',
                 'Maimay', 'Balkhab', 'Kang', 'Kiran Wa Menjan', 'Kiti', 'Bangi', 'Kakar', 'Rustaq',
                 'Chal', 'Dawlatyar', 'Bamyan', 'Khas Kunar', 'Andarab', 'Hissa-e-Duwumi Kohistan',
                 'Farkhar', 'Khwahan', 'Vakhsh', 'Nesay', 'Khedir', 'Deh Salah', 'Pul-e-Hisar',
                 'Hazar Sumuch', 'Ishterlai', 'Paryan', 'Unaba', 'Feroz Nakhcheer', 'Firing Wa Gharu',
                 'Mandol', 'Shiki', 'Wakhan', 'Shighnan', 'Surkhi Parsa']

    # Combine all locations
    all_locations = provinces + districts

    # Sort by length (longest first) to avoid partial replacements
    all_locations.sort(key=len, reverse=True)

    masked_text = text
    for location in all_locations:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(location) + r'\b'
        masked_text = re.sub(pattern, '', masked_text, flags=re.IGNORECASE)

    # Clean up multiple spaces and extra whitespace
    masked_text = re.sub(r'\s+', ' ', masked_text).strip()

    return masked_text


def get_embedding_model():
    """Lazy load the embedding model"""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print("Loading sentence transformer model...")
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
    return EMBEDDING_MODEL


def create_text_embeddings(df, text_columns, batch_size=32, mask_entities=True, mask_locations=True):
    """
    Create embeddings for specified text columns

    Args:
        df: DataFrame containing text columns
        text_columns: List of column names to embed (e.g., ['notes', 'actor2'])
        batch_size: Batch size for encoding
        mask_entities: Whether to remove group names before embedding (default: True)
        mask_locations: Whether to remove location names before embedding (default: True)

    Returns:
        DataFrame with embedding columns added
    """
    model = get_embedding_model()
    df_copy = df.copy()

    for col in text_columns:
        if col not in df_copy.columns:
            print(f"Warning: Column '{col}' not found in DataFrame")
            continue

        print(f"Creating embeddings for column: {col}")

        # Handle missing values
        text_data = df_copy[col].fillna('').astype(str)

        # Remove group names if requested
        if mask_entities:
            print(f"  Removing group names from '{col}'...")
            text_data = text_data.apply(mask_group_names)

        # Remove location names if requested
        if mask_locations:
            print(f"  Removing location names from '{col}'...")
            text_data = text_data.apply(mask_location_names)

        text_data = text_data.tolist()

        # Create embeddings
        embeddings = model.encode(
            text_data,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Add embeddings as new columns
        embedding_dim = embeddings.shape[1]
        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f'{col}_emb_{i}' for i in range(embedding_dim)],
            index=df_copy.index
        )

        # Concatenate with original dataframe
        df_copy = pd.concat([df_copy, embedding_df], axis=1)

        print(f"Added {embedding_dim} embedding features for '{col}'")

    return df_copy


# ============================================================================
# LEAN FEATURES: Only Numeric-Derived and Aggregate/Count Features
# ============================================================================

def create_aggregate_attack_features(df, text_column='notes'):
    """
    Create aggregate attack count features from text

    Args:
        df: Input DataFrame
        text_column: Name of text column to analyze

    Returns:
        DataFrame with attack count columns added
    """

    def count_attack_signals(text):
        """Count number of direct attack indicators in text"""
        if not isinstance(text, str):
            return 0

        text_lower = text.lower()

        # Attack signal terms
        attack_signals = [
            'suicide', 'bomber',
            'executed', 'beheaded', 'stoned to death',
            'abducted', 'kidnapped', 'captured',
            'targeted', 'assassination', 'assassinated'
        ]

        # Count how many signals are present
        count = sum(1 for signal in attack_signals if signal in text_lower)
        return count

    if text_column not in df.columns:
        print(f"Warning: Column '{text_column}' not found. Skipping attack count features.")
        return df

    print(f"Creating attack count features from '{text_column}'...")

    df_copy = df.copy()

    # Create count feature
    df_copy['direct_attack_count'] = df_copy[text_column].apply(count_attack_signals)

    # Create binary flag for strong signal (2+ indicators)
    df_copy['has_strong_attack_signal'] = (df_copy['direct_attack_count'] >= 2).astype(int)

    print(
        f"  Added 'direct_attack_count' (range: {df_copy['direct_attack_count'].min()}-{df_copy['direct_attack_count'].max()})")
    print(f"  Added 'has_strong_attack_signal' ({df_copy['has_strong_attack_signal'].sum()} samples with 2+ signals)")

    return df_copy


def create_casualty_threshold_features(df, fatalities_column='fatalities'):
    """
    Create threshold-based features from fatalities column

    Args:
        df: Input DataFrame
        fatalities_column: Name of fatalities column

    Returns:
        DataFrame with casualty threshold columns added
    """
    if fatalities_column not in df.columns:
        print(f"Warning: Column '{fatalities_column}' not found. Skipping casualty features.")
        return df

    print(f"Creating casualty threshold features from '{fatalities_column}'...")

    df_copy = df.copy()

    # Create threshold features
    df_copy['has_casualties'] = (df_copy[fatalities_column] > 0).astype(int)
    df_copy['high_casualties'] = (df_copy[fatalities_column] >= 10).astype(int)
    df_copy['very_high_casualties'] = (df_copy[fatalities_column] >= 20).astype(int)
    df_copy['zero_fatalities'] = (df_copy[fatalities_column] == 0).astype(int)

    # Print statistics
    print(f"  Added 'has_casualties' ({df_copy['has_casualties'].sum()} samples)")
    print(f"  Added 'high_casualties' ({df_copy['high_casualties'].sum()} samples with ≥10 deaths)")
    print(f"  Added 'very_high_casualties' ({df_copy['very_high_casualties'].sum()} samples with ≥20 deaths)")
    print(f"  Added 'zero_fatalities' ({df_copy['zero_fatalities'].sum()} samples)")

    return df_copy






# ---------Feature Eng Old--------

def feature_creating(df, use_embeddings=True, text_columns=None, use_lean_features=True):
    """
    Feature engineering pipeline with optional text embeddings and lean features

    Args:
        df: Input DataFrame
        use_embeddings: Whether to create text embeddings (default: True)
        text_columns: List of text columns to embed (default: ['notes', 'actor2'])
                     Set to None to skip embeddings
        use_lean_features: Whether to create lean context features (default: True)
                          Adds 6 features: 4 casualty thresholds + 2 attack counts

    Returns:
        working_df: Processed DataFrame with features
        unattrib_df: DataFrame of unattributed attacks
    """
    # -----Cols to Drop----
    to_drop = [
        'event_id_cnty',
        'disorder_type',
        'time_precision',
        'source',
        'source_scale',
        'iso',
        'country',
        'timestamp',
        'geo_precision',
        'year',
        'region',
        'latitude',
        'longitude',
        'interaction',
        'event_date']
    # -----Cols to Drop----

    # ------Groups-----
    unattrib = [
        'Unidentified Armed Group (Afghanistan)',
        'Taliban and/or Islamic State Khorasan Province (ISKP)'
    ]

    taliban = [
        'Taliban',
        'Taliban - Red Unit',
        'Mutiny of Taliban'
    ]

    iskp = [
        'Islamic State Khorasan Province (ISKP)'
    ]

    other = [
        'Al Qaeda',
        'HQN: Haqqani Network',
        'TTP: Tehreek-i-Taliban Pakistan',
        'LeI: Lashkar-e-Islam'
    ]
    # ------Groups-----

    # -----Tags for 'violence against women tags'----
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
    # -----Tags for 'violence against women tags'----

    # df = df.drop(columns=to_drop)
    # df['event_date'] = pd.to_datetime(df['event_date'])
    # df = df.sort_values(by='event_date', ascending=True)
    # df = df.reset_index(drop=True)

    # create a dataframe of unattributed attacks
    # to be used later
    unattrib_df = df[df['actor1'].isin(unattrib)].copy()
    unattrib_df = unattrib_df.reset_index(drop=True)

    # mapping the target vars
    mapping = {name: 0 for name in taliban}
    mapping.update({name: 1 for name in iskp})

    working_df = df.copy()

    # map to the df
    working_df = working_df[working_df['actor1'].isin(taliban + iskp)].copy()
    working_df['target'] = working_df['actor1'].map(mapping).astype(int)

    # create violence against women tags
    working_df['violence_against_women'] = (
            (working_df['sub_event_type'] == 'Sexual violence') |
            ((working_df['tags'].isin(violence_against_women_tags)) & (working_df['tags'] != 'local administrators'))
    ).astype(int)

    working_df = working_df.reset_index(drop=True)
    col = "civilian_targeting"

    working_df[col] = working_df[col].notna().astype(int)

    working_df = working_df.drop(index=[2300, 28966, 31115])
    working_df = working_df.reset_index(drop=True)

    # -----ADD LEAN FEATURES (Numeric-Derived + Aggregate/Count)-----
    if use_lean_features:
        print("\n" + "=" * 60)
        print("CREATING LEAN FEATURES (6 total)")
        print("=" * 60)

        # 1. Casualty threshold features (4 features)
        working_df = create_casualty_threshold_features(working_df, fatalities_column='fatalities')

        # 2. Aggregate attack count features (2 features)
        working_df = create_aggregate_attack_features(working_df, text_column='notes')

        print("=" * 60)
        print("Lean features created successfully!")
        print("=" * 60 + "\n")

    # -----ADD TEXT EMBEDDINGS-----
    if use_embeddings and text_columns is not None:
        print("\n" + "=" * 60)
        print("CREATING TEXT EMBEDDINGS")
        print("=" * 60)

        # Create embeddings for specified text columns
        working_df = create_text_embeddings(working_df, text_columns)

        print("=" * 60)
        print(f"Total features after embeddings: {working_df.shape[1]}")
        print("=" * 60 + "\n")

    # -----ONE-HOT ENCODING (after all feature creation)-----
    encoded_cols = ['sub_event_type']
    working_df = pd.get_dummies(working_df, columns=encoded_cols, dtype=int)

    return working_df, unattrib_df

# ---------Feature Eng Old--------




