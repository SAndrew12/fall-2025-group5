import streamlit as st
import pandas as pd
import numpy as np
import pathlib
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import re
import joblib  # <-- for loading joblib-saved models & scalers

warnings.filterwarnings('ignore')

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SAVED_MODELS_DIR = PROJECT_ROOT / "src" / "saved_models"
DEMO_DIR = PROJECT_ROOT / "demo"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


# ============================================================================
# Text Preprocessing Functions
# ============================================================================

def mask_group_names(text):
    """Remove all variants of Taliban and ISIS-K group names"""
    if not isinstance(text, str):
        return text

    taliban_variants = [
        r'\bTaliban\b', r'\bTaleban\b', r'\bTaliban-e\b', r'\bT-Ban\b',
        r'\bTban\b', r'\bTTP\b', r'\bTehrik-i-Taliban\b', r'\bTehreek-e-Taliban\b',
    ]

    isis_variants = [
        r'\bISIS-K\b', r'\bISIS-KP\b', r'\bISIS Khorasan\b', r'\bISIS-Khorasan\b',
        r'\bISIL-K\b', r'\bISIL Khorasan\b', r'\bIS-K\b', r'\bIS-KP\b',
        r'\bIslamic State Khorasan\b', r'\bIslamic State in Khorasan\b',
        r'\bIslamic State of Khorasan\b', r'\bDaesh\b', r"\bDa\'esh\b",
        r'\bISIS\b', r'\bISIL\b', r'\bIslamic State\b', r'\bIS\b'
    ]

    all_variants = taliban_variants + isis_variants
    masked_text = text

    for pattern in all_variants:
        masked_text = re.sub(pattern, '', masked_text, flags=re.IGNORECASE)

    masked_text = re.sub(r'\s+', ' ', masked_text).strip()
    return masked_text


def mask_location_names(text):
    """Remove all location names (provinces and districts) from text"""
    if not isinstance(text, str):
        return text

    provinces = ['Wardak', 'Shewa', 'Farah', 'Helmand', 'Kandahar', 'Nimruz', 'Paktia', 'Paktika',
                 'Parwan', 'Logar', 'Kunar', 'Ghazni', 'Laghman', 'Herat', 'Kabul', 'Sar-e Pol',
                 'Nangarhar', 'Nangahar', 'Khost', 'Zabul', 'Badakhshan', 'Faryab', 'Kunduz', 'Badghis',
                 'Balkh', 'Kapisa', 'Baghlan', 'Samangan', 'Urozgan', 'Jowzjan', 'Daykundi',
                 'Nuristan', 'Takhar', 'Ghor', 'Bamyan', 'Panjshir', 'Khatlon']

    districts = ['Sayyid Abad', 'Bala Buluk', 'Lashkargah', 'Nahr-i-Saraj', 'Maruf', 'Khashrod',
                 'Zurmat', 'Sar Rawza', 'Bagram', 'Khak-i-Safed', 'Baraki Barak', 'Sangin',
                 'Sar Kani', 'Waghaz', 'Charkh', 'Dawlat Shah', 'Kushk-i-Kuhna', 'Surubi',
                 'Herat', 'Sancharak', 'Shinwar', 'Ghazi Abad', 'Andar', 'Nawa-i-Barikzayi',
                 'Nad Ali', 'Mata Khan', 'Puli Alam', 'Khost', 'Shemel Zayi', 'Gelan', 'Maiwand',
                 'Ghazni', 'Bati Kot', 'Faiz Abad', 'Gardez', 'Almar', 'Dangam', 'Rodat', 'Nesh',
                 'Spin Boldak', 'Mohammad Agha', 'Farah', 'Khan Abad', 'Shindand', 'Pushtrud',
                 'Jalalabad', 'Mazar-e-Sharif', 'Kandahar', 'Kunduz', 'Kabul']

    all_locations = provinces + districts
    all_locations.sort(key=len, reverse=True)

    masked_text = text
    for location in all_locations:
        pattern = r'\b' + re.escape(location) + r'\b'
        masked_text = re.sub(pattern, '', masked_text, flags=re.IGNORECASE)

    masked_text = re.sub(r'\s+', ' ', masked_text).strip()
    return masked_text


def preprocess_notes(text):
    """Apply all preprocessing to notes text"""
    text = mask_group_names(text)
    text = mask_location_names(text)
    return text


# ============================================================================
# Model Architecture Definitions
# ============================================================================

class TextDataset(Dataset):
    """Dataset for BERT text classification"""

    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


class TextFeatureDataset(Dataset):
    """Dataset for BERT-Hybrid (text + manual features)"""

    def __init__(self, texts, manual_features, tokenizer, max_length=320):
        self.texts = texts
        self.manual_features = manual_features
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        manual_feat = self.manual_features[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'manual_features': torch.tensor(manual_feat, dtype=torch.float)
        }


class BERTWithManualFeatures(nn.Module):
    """BERT model with feature fusion for BERT-Hybrid"""

    def __init__(self, bert_model_name='bert-base-uncased',
                 num_manual_features=14,
                 hidden_dim=128,
                 dropout=0.3):
        super(BERTWithManualFeatures, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.feature_layer = nn.Linear(num_manual_features, hidden_dim)

        bert_hidden_size = self.bert.config.hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(bert_hidden_size + hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(128, 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, manual_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]

        feature_embedding = self.relu(self.feature_layer(manual_features))
        feature_embedding = self.dropout(feature_embedding)

        combined = torch.cat([cls_embedding, feature_embedding], dim=1)
        x = self.fusion(combined)
        logits = self.classifier(x)
        return logits


# ============================================================================
# Data Processing Functions
# ============================================================================

def load_and_filter_data(csv_path):
    """Load CSV and filter to Taliban/ISIS-K attacks only"""
    df = pd.read_csv(csv_path)
    #filtered_df = df[df['actor1'] == 'Taliban and/or Islamic State Khorasan Province (ISKP)'].copy()
    filtered_df = df
    # Apply text preprocessing to notes column
    filtered_df['notes_processed'] = filtered_df['notes'].apply(preprocess_notes)

    return filtered_df


def prepare_data_for_model(df, model_type, use_processed_notes=True):
    """Prepare data based on model type"""
    notes_col = 'notes_processed' if use_processed_notes else 'notes'

    if model_type == "Classical":
        feature_cols = MANUAL_FEATURES + EMBEDDING_FEATURES
        return df[feature_cols]

    elif model_type == "BERT":
        return df[[notes_col]]

    elif model_type == "BERT-Hybrid":
        feature_cols = MANUAL_FEATURES + [notes_col]
        return df[feature_cols]

    return df


# ============================================================================
# Helper for preprocessors (classical models)
# ============================================================================

def apply_preprocessors(preprocessor, X):
    """
    Apply one or more preprocessors to X.
    Handles:
      - single transformer with .transform()
      - list/tuple of transformers
      - dict of transformers
    """
    if preprocessor is None:
        return X

    X_processed = X

    # Single object with transform()
    if hasattr(preprocessor, "transform"):
        return preprocessor.transform(X_processed)

    # List / tuple of preprocessors
    if isinstance(preprocessor, (list, tuple)):
        for p in preprocessor:
            if hasattr(p, "transform"):
                X_processed = p.transform(X_processed)
        return X_processed

    # Dict of preprocessors
    if isinstance(preprocessor, dict):
        for p in preprocessor.values():
            if hasattr(p, "transform"):
                X_processed = p.transform(X_processed)
        return X_processed

    # Fallback: unknown structure, just return X
    return X_processed


# ============================================================================
# Model Loading Functions
# ============================================================================

@st.cache_resource
def load_classical_model(model_path, preprocessor_path):
    """Load a classical ML model and its preprocessor saved with joblib."""
    try:
        loaded_model = joblib.load(model_path)

        # Check if the model file contains a tuple (model, preprocessor)
        if isinstance(loaded_model, tuple):
            model = loaded_model[0]
            st.info(" Model file contained tuple - extracted model object")
        else:
            model = loaded_model

    except Exception as e:
        raise Exception(f"Could not load model from {model_path}. Error: {str(e)}")

    try:
        preprocessor = joblib.load(preprocessor_path)
    except Exception as e:
        raise Exception(f"Could not load preprocessor from {preprocessor_path}. Error: {str(e)}")

    return model, preprocessor


@st.cache_resource
def load_bert_model(model_dir):
    """Load BERT model"""
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_bert_hybrid_model(model_dir):
    """Load BERT-Hybrid (Feature Fusion) model"""
    checkpoint = torch.load(f"{model_dir}/model.pt", map_location=device)
    num_manual_features = checkpoint.get('num_manual_features', 14)
    prediction_threshold = checkpoint.get('prediction_threshold', 0.5)

    model = BERTWithManualFeatures(
        bert_model_name='bert-base-uncased',
        num_manual_features=num_manual_features,
        hidden_dim=128,
        dropout=0.3
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # Load scaler (saved with joblib in save_trained_models.py)
    scaler_path = pathlib.Path(model_dir) / "feature_scaler.pkl"
    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            st.success("Feature scaler loaded successfully")
        except Exception as e:
            st.error(f"Could not load scaler from {scaler_path}. Predictions may be incorrect.")
            st.error(f"Error: {str(e)}")
            scaler = None
    else:
        st.error(f"Scaler file not found: {scaler_path}")
        st.error("The model was trained WITH feature scaling. Predictions will be incorrect without the scaler!")

    return model, tokenizer, scaler, prediction_threshold


# ============================================================================
# Batch Prediction Functions
# ============================================================================

def predict_classical(model, preprocessor, X):
    """Make predictions with classical model"""
    # Keep X as DataFrame so column-based preprocessors still work
    X_processed = apply_preprocessors(preprocessor, X)
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed) if hasattr(model, 'predict_proba') else None
    return predictions, probabilities


def predict_bert(model, tokenizer, texts, batch_size=16, max_length=128):
    """Make predictions with BERT model"""
    dataset = TextDataset(texts.tolist(), tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    return np.array(predictions), np.array(probabilities)


def predict_bert_hybrid(model, tokenizer, scaler, texts, manual_features,
                        batch_size=16, max_length=320, threshold=0.5):
    """Make predictions with BERT-Hybrid model"""
    # Apply the same scaling used during training
    if scaler is not None:
        manual_features = scaler.transform(manual_features)
    else:
        st.warning(" No scaler available - predictions may be incorrect!")

    dataset = TextFeatureDataset(
        texts.tolist(),
        manual_features,
        tokenizer,
        max_length
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            manual_feat = batch['manual_features'].to(device)

            logits = model(input_ids, attention_mask, manual_feat)
            probs = torch.softmax(logits, dim=1)
            preds = (probs[:, 1] >= threshold).long()

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    return np.array(predictions), np.array(probabilities)


# ============================================================================
# Single Event Prediction Functions
# ============================================================================

def predict_single_classical(model, preprocessor, X_single):
    """Make prediction for a single event with classical model"""
    # X_single should be a DataFrame with one row
    X_processed = apply_preprocessors(preprocessor, X_single)
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0] if hasattr(model, 'predict_proba') else None
    return prediction, probability


def predict_single_bert(model, tokenizer, text, max_length=128):
    """Make prediction for a single event with BERT model"""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        prediction = pred.cpu().numpy()[0]
        probability = probs.cpu().numpy()[0]

    return prediction, probability


def predict_single_bert_hybrid(model, tokenizer, scaler, text, manual_features,
                               max_length=320, threshold=0.5):
    """Make prediction for a single event with BERT-Hybrid model"""
    # Scale manual features if scaler is provided
    if scaler is not None:
        manual_features_scaled = scaler.transform(manual_features.reshape(1, -1))
    else:
        manual_features_scaled = manual_features.reshape(1, -1)

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        manual_feat = torch.tensor(manual_features_scaled, dtype=torch.float).to(device)

        logits = model(input_ids, attention_mask, manual_feat)
        probs = torch.softmax(logits, dim=1)
        pred = (probs[:, 1] >= threshold).long()

        prediction = pred.cpu().numpy()[0]
        probability = probs.cpu().numpy()[0]

    return prediction, probability


# ============================================================================
# Streamlit App
# ============================================================================

def main():
    st.set_page_config(page_title="Taliban vs ISIS-K Classifier", layout="wide")

    st.title("Taliban vs ISIS-K Attack Classifier")

    st.markdown("""
    This demo classifies conflict events using three different approaches:
    - **Classical ML**: Random Forest, XGBoost, MLP with manual features + embeddings
    - **BERT**: Fine-tuned BERT using preprocessed event descriptions
    - **BERT-Hybrid**: Feature fusion combining BERT with manual features

    **Text Preprocessing**: Group names (Taliban, ISIS-K) and location names are removed before classification.
    """)

    # ---- Sidebar: Model Selection ----
    with st.sidebar:
        st.header(" Configuration")

        model_type = st.selectbox(
            "Model Type:",
            ["Classical", "BERT", "BERT-Hybrid"],
            help="Select which modeling approach to use"
        )

        # Map model type to directory name
        dir_mapping = {
            'classical': 'classical',
            'bert': 'bert',
            'bert-hybrid': 'feature_fusion'  # BERT-Hybrid models are in feature_fusion directory
        }
        model_dir = SAVED_MODELS_DIR / dir_mapping[model_type.lower()]

        if model_type == "Classical":
            if model_dir.exists():
                model_files = [f.name for f in model_dir.iterdir()
                               if f.suffix == '.pkl' and 'preprocessor' not in f.name]
                selected_model = st.selectbox("Select Model:", model_files)
            else:
                st.error(f"Model directory not found: {model_dir}")
                return
        else:
            selected_model = "default"

        st.divider()
        st.markdown(f"**Model Type:** {model_type}")
        st.markdown(f"**Device:** {device}")

    # ---- Load Data ----
    st.header(" Data")

    sample_csv = DEMO_DIR / "unattributed_attacks_processed.csv"

    if not sample_csv.exists():
        st.error(f"Data file not found: {sample_csv}")
        return

    # Load and filter data
    with st.spinner("Loading data..."):
        df = load_and_filter_data(sample_csv)

    st.success(f"Loaded **{len(df)}** Taliban/ISIS-K attacks for classification")

    # Prepare data based on model type
    prepared_data = prepare_data_for_model(df, model_type)

    # Show data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(prepared_data))
    with col2:
        if model_type == "BERT":
            st.metric("Features", "Text Only")
        else:
            st.metric("Features", len(prepared_data.columns))
    with col3:
        st.metric("Model Type", model_type)

    # Show data preview
    with st.expander(" View Data Preview (first 10 rows)"):
        if model_type == "BERT":
            preview_df = pd.DataFrame({
                'event_id': df['event_id_cnty'].head(10).values,
                'date': df['event_date'].head(10).values,
                'location': df['location'].head(10).values,
                'notes_preprocessed': prepared_data['notes_processed'].head(10).values
            })
            st.dataframe(preview_df, use_container_width=True, height=300)
        else:
            st.dataframe(prepared_data.head(10), use_container_width=True, height=300)

    # ---- Prediction Tabs ----
    tab1, tab2 = st.tabs(["Single Event Classification", " Batch Predictions"])

    # ====================
    # TAB 1: Single Event
    # ====================
    with tab1:
        st.header(" Single Event Classification")
        st.markdown("Select an individual event to classify")

        # Event selection
        event_options = [f"{row['event_id_cnty']} - {row['event_date']} - {row['location']}"
                         for idx, row in df.iterrows()]
        selected_event_str = st.selectbox(
            "Select Event:",
            event_options,
            help="Choose an event to classify"
        )

        # Get the selected event index
        selected_idx = event_options.index(selected_event_str)
        selected_row = df.iloc[selected_idx]

        # Display event details
        st.subheader(" Event Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Event ID", selected_row['event_id_cnty'])
        with col2:
            st.metric("Date", selected_row['event_date'])
        with col3:
            st.metric("Fatalities", int(selected_row['fatalities']))

        st.markdown(f"**Location:** {selected_row['location']}")
        st.markdown("**Event Description:**")
        st.info(selected_row['notes'])

        # Classify button
        if st.button(" Classify This Event", type="primary", use_container_width=True, key="single_classify"):
            try:
                with st.spinner(f"Loading {model_type} model..."):
                    if model_type == "Classical":
                        model_path = model_dir / selected_model
                        preprocessor_path = model_dir / "preprocessors.pkl"
                        model, preprocessor = load_classical_model(model_path, preprocessor_path)

                    elif model_type == "BERT":
                        model, tokenizer = load_bert_model(model_dir)

                    elif model_type == "BERT-Hybrid":
                        model, tokenizer, scaler, threshold = load_bert_hybrid_model(model_dir)

                st.success(" Model loaded successfully!")

                # Make prediction
                with st.spinner("Classifying event..."):
                    if model_type == "Classical":
                        X_single = prepared_data.iloc[[selected_idx]]
                        prediction, probability = predict_single_classical(model, preprocessor, X_single)

                    elif model_type == "BERT":
                        text = prepared_data.iloc[selected_idx]['notes_processed']
                        prediction, probability = predict_single_bert(model, tokenizer, text)

                    elif model_type == "BERT-Hybrid":
                        text = prepared_data.iloc[selected_idx]['notes_processed']
                        manual_feats = prepared_data.iloc[selected_idx][MANUAL_FEATURES].values
                        prediction, probability = predict_single_bert_hybrid(
                            model, tokenizer, scaler, text, manual_feats, threshold=threshold
                        )

                # Display prediction
                st.success(" Classification complete!")

                st.subheader(" Prediction Results")

                prediction_label = "Taliban" if prediction == 0 else "ISIS-K"

                # Color-coded prediction
                if prediction == 0:
                    st.success(f"### Predicted: **{prediction_label}**")
                else:
                    st.error(f"### Predicted: **{prediction_label}**")

                # Show probabilities
                if probability is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Taliban Probability",
                            f"{probability[0]:.1%}",
                            help="Confidence that this is a Taliban attack"
                        )
                    with col2:
                        st.metric(
                            "ISIS-K Probability",
                            f"{probability[1]:.1%}",
                            help="Confidence that this is an ISIS-K attack"
                        )

                    # Confidence bar
                    st.markdown("**Confidence:**")
                    confidence = float(max(probability))
                    st.progress(confidence)
                    st.caption(f"{confidence:.1%} confidence in prediction")

            except Exception as e:
                st.error(f" Error during prediction: {str(e)}")
                st.exception(e)

    # ====================
    # TAB 2: Batch Predictions
    # ====================
    with tab2:
        st.header(" Batch Predictions")
        st.markdown("Classify all events at once")

        if st.button(" Run Predictions on All Events", type="primary", use_container_width=True, key="batch_classify"):
            try:
                with st.spinner(f"Loading {model_type} model..."):
                    if model_type == "Classical":
                        model_path = model_dir / selected_model
                        preprocessor_path = model_dir / "preprocessors.pkl"
                        model, preprocessor = load_classical_model(model_path, preprocessor_path)

                    elif model_type == "BERT":
                        model, tokenizer = load_bert_model(model_dir)

                    elif model_type == "BERT-Hybrid":
                        model, tokenizer, scaler, threshold = load_bert_hybrid_model(model_dir)

                st.success(" Model loaded successfully!")

                # Make predictions
                with st.spinner(f"Making predictions on {len(df)} events..."):
                    # Sanity checks
                    if model_type == "Classical":
                        if not (model_dir / selected_model).exists():
                            st.error(f"Model file not found: {model_dir / selected_model}")
                            st.stop()
                        if not (model_dir / "preprocessors.pkl").exists():
                            st.error(f"Preprocessor file not found: {model_dir / 'preprocessors.pkl'}")
                            st.stop()

                    elif model_type == "BERT-Hybrid":
                        if not (model_dir / "model.pt").exists():
                            st.error(f"Model file not found: {model_dir / 'model.pt'}")
                            st.info(f"Looking in directory: {model_dir}")
                            st.info(
                                f"Files available: {list(model_dir.iterdir()) if model_dir.exists() else 'Directory does not exist'}")
                            st.stop()

                    # Now make predictions
                    if model_type == "Classical":
                        X = prepared_data  # keep as DataFrame for preprocessors
                        predictions, probabilities = predict_classical(model, preprocessor, X)

                    elif model_type == "BERT":
                        texts = prepared_data['notes_processed']
                        predictions, probabilities = predict_bert(model, tokenizer, texts)

                    elif model_type == "BERT-Hybrid":
                        texts = prepared_data['notes_processed']
                        manual_feats = prepared_data[MANUAL_FEATURES].values
                        predictions, probabilities = predict_bert_hybrid(
                            model, tokenizer, scaler, texts, manual_feats, threshold=threshold
                        )

                st.success("Predictions complete!")

                # Display results
                st.subheader(" Results Summary")

                # Class distribution
                pred_counts = pd.Series(predictions).value_counts()
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Predicted Taliban (Class 0)",
                              int(pred_counts.get(0, 0)),
                              delta=f"{pred_counts.get(0, 0) / len(predictions) * 100:.1f}%")

                with col2:
                    st.metric("Predicted ISIS-K (Class 1)",
                              int(pred_counts.get(1, 0)),
                              delta=f"{pred_counts.get(1, 0) / len(predictions) * 100:.1f}%")

                # Create results dataframe
                results_df = df[['event_id_cnty', 'event_date', 'location', 'fatalities', 'notes']].copy()
                results_df['prediction'] = predictions
                results_df['prediction_label'] = results_df['prediction'].map({0: 'Taliban', 1: 'ISIS-K'})

                if probabilities is not None:
                    results_df['prob_taliban'] = probabilities[:, 0]
                    results_df['prob_isis_k'] = probabilities[:, 1]
                    results_df['confidence'] = np.max(probabilities, axis=1)

                # Display results table
                st.subheader(" Detailed Results")

                display_cols = ['event_id_cnty', 'event_date', 'location', 'fatalities',
                                'prediction_label', 'confidence', 'prob_taliban',
                                'prob_isis_k'] if probabilities is not None else \
                    ['event_id_cnty', 'event_date', 'location', 'fatalities', 'prediction_label']

                st.dataframe(
                    results_df[display_cols],
                    use_container_width=True,
                    height=400
                )

                # Show sample predictions with notes
                with st.expander(" View Sample Predictions with Event Descriptions"):
                    sample_size = min(10, len(results_df))
                    sample_df = results_df[['event_id_cnty', 'event_date', 'location', 'notes',
                                            'prediction_label', 'confidence']].head(sample_size)

                    for idx, row in sample_df.iterrows():
                        st.markdown(f"**{row['event_id_cnty']}** - {row['event_date']} - {row['location']}")
                        st.write(f"*{row['notes'][:200]}...*")
                        st.markdown(
                            f"**Prediction:** {row['prediction_label']} ({row['confidence'] * 100:.1f}% confidence)")
                        st.divider()

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Full Results as CSV",
                    data=csv,
                    file_name=f"predictions_{model_type.lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f" Error during prediction: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# import pathlib
# import pickle
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, BertModel
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# import warnings
# import re
# import joblib
#
# warnings.filterwarnings('ignore')
#
# PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
# SAVED_MODELS_DIR = PROJECT_ROOT / "src" / "saved_models"
# DEMO_DIR = PROJECT_ROOT / "demo"
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Define feature sets
# MANUAL_FEATURES = [
#     'civilian_targeting',
#     'fatalities',
#     'violence_against_women',
#     'sub_event_type_Abduction/forced disappearance',
#     'sub_event_type_Air/drone strike',
#     'sub_event_type_Armed clash',
#     'sub_event_type_Attack',
#     'sub_event_type_Government regains territory',
#     'sub_event_type_Grenade',
#     'sub_event_type_Non-state actor overtakes territory',
#     'sub_event_type_Remote explosive/landmine/IED',
#     'sub_event_type_Sexual violence',
#     'sub_event_type_Shelling/artillery/missile attack',
#     'sub_event_type_Suicide bomb'
# ]
#
# EMBEDDING_FEATURES = [f'notes_emb_{i}' for i in range(384)]
#
#
# # ============================================================================
# # Text Preprocessing Functions
# # ============================================================================
#
# def mask_group_names(text):
#     """Remove all variants of Taliban and ISIS-K group names"""
#     if not isinstance(text, str):
#         return text
#
#     taliban_variants = [
#         r'\bTaliban\b', r'\bTaleban\b', r'\bTaliban-e\b', r'\bT-Ban\b',
#         r'\bTban\b', r'\bTTP\b', r'\bTehrik-i-Taliban\b', r'\bTehreek-e-Taliban\b',
#     ]
#
#     isis_variants = [
#         r'\bISIS-K\b', r'\bISIS-KP\b', r'\bISIS Khorasan\b', r'\bISIS-Khorasan\b',
#         r'\bISIL-K\b', r'\bISIL Khorasan\b', r'\bIS-K\b', r'\bIS-KP\b',
#         r'\bIslamic State Khorasan\b', r'\bIslamic State in Khorasan\b',
#         r'\bIslamic State of Khorasan\b', r'\bDaesh\b', r"\bDa\'esh\b",
#         r'\bISIS\b', r'\bISIL\b', r'\bIslamic State\b', r'\bIS\b'
#     ]
#
#     all_variants = taliban_variants + isis_variants
#     masked_text = text
#
#     for pattern in all_variants:
#         masked_text = re.sub(pattern, '', masked_text, flags=re.IGNORECASE)
#
#     masked_text = re.sub(r'\s+', ' ', masked_text).strip()
#     return masked_text
#
#
# def mask_location_names(text):
#     """Remove all location names (provinces and districts) from text"""
#     if not isinstance(text, str):
#         return text
#
#     provinces = ['Wardak', 'Shewa', 'Farah', 'Helmand', 'Kandahar', 'Nimruz', 'Paktia', 'Paktika',
#                  'Parwan', 'Logar', 'Kunar', 'Ghazni', 'Laghman', 'Herat', 'Kabul', 'Sar-e Pol',
#                  'Nangarhar', 'Nangahar', 'Khost', 'Zabul', 'Badakhshan', 'Faryab', 'Kunduz', 'Badghis',
#                  'Balkh', 'Kapisa', 'Baghlan', 'Samangan', 'Urozgan', 'Jowzjan', 'Daykundi',
#                  'Nuristan', 'Takhar', 'Ghor', 'Bamyan', 'Panjshir', 'Khatlon']
#
#     districts = ['Sayyid Abad', 'Bala Buluk', 'Lashkargah', 'Nahr-i-Saraj', 'Maruf', 'Khashrod',
#                  'Zurmat', 'Sar Rawza', 'Bagram', 'Khak-i-Safed', 'Baraki Barak', 'Sangin',
#                  'Sar Kani', 'Waghaz', 'Charkh', 'Dawlat Shah', 'Kushk-i-Kuhna', 'Surubi',
#                  'Herat', 'Sancharak', 'Shinwar', 'Ghazi Abad', 'Andar', 'Nawa-i-Barikzayi',
#                  'Nad Ali', 'Mata Khan', 'Puli Alam', 'Khost', 'Shemel Zayi', 'Gelan', 'Maiwand',
#                  'Ghazni', 'Bati Kot', 'Faiz Abad', 'Gardez', 'Almar', 'Dangam', 'Rodat', 'Nesh',
#                  'Spin Boldak', 'Mohammad Agha', 'Farah', 'Khan Abad', 'Shindand', 'Pushtrud',
#                  'Jalalabad', 'Mazar-e-Sharif', 'Kandahar', 'Kunduz', 'Kabul']
#
#     all_locations = provinces + districts
#     all_locations.sort(key=len, reverse=True)
#
#     masked_text = text
#     for location in all_locations:
#         pattern = r'\b' + re.escape(location) + r'\b'
#         masked_text = re.sub(pattern, '', masked_text, flags=re.IGNORECASE)
#
#     masked_text = re.sub(r'\s+', ' ', masked_text).strip()
#     return masked_text
#
#
# def preprocess_notes(text):
#     """Apply all preprocessing to notes text"""
#     text = mask_group_names(text)
#     text = mask_location_names(text)
#     return text
#
#
# # ============================================================================
# # Model Architecture Definitions
# # ============================================================================
#
# class TextDataset(Dataset):
#     """Dataset for BERT text classification"""
#
#     def __init__(self, texts, tokenizer, max_length=128):
#         self.texts = texts
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten()
#         }
#
#
# class TextFeatureDataset(Dataset):
#     """Dataset for BERT-Hybrid (text + manual features)"""
#
#     def __init__(self, texts, manual_features, tokenizer, max_length=320):
#         self.texts = texts
#         self.manual_features = manual_features
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         manual_feat = self.manual_features[idx]
#
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'manual_features': torch.tensor(manual_feat, dtype=torch.float)
#         }
#
#
# class BERTWithManualFeatures(nn.Module):
#     """BERT model with feature fusion for BERT-Hybrid"""
#
#     def __init__(self, bert_model_name='bert-base-uncased',
#                  num_manual_features=14,
#                  hidden_dim=128,
#                  dropout=0.3):
#         super(BERTWithManualFeatures, self).__init__()
#
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.feature_layer = nn.Linear(num_manual_features, hidden_dim)
#
#         bert_hidden_size = self.bert.config.hidden_size
#         self.fusion = nn.Sequential(
#             nn.Linear(bert_hidden_size + hidden_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#         )
#
#         self.classifier = nn.Linear(128, 2)
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()
#
#     def forward(self, input_ids, attention_mask, manual_features):
#         bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         cls_embedding = bert_output.last_hidden_state[:, 0, :]
#
#         feature_embedding = self.relu(self.feature_layer(manual_features))
#         feature_embedding = self.dropout(feature_embedding)
#
#         combined = torch.cat([cls_embedding, feature_embedding], dim=1)
#         x = self.fusion(combined)
#         logits = self.classifier(x)
#         return logits
#
#
# # ============================================================================
# # Data Processing Functions
# # ============================================================================
#
# def load_and_filter_data(csv_path):
#     """Load CSV and filter to Taliban/ISIS-K attacks only"""
#     df = pd.read_csv(csv_path)
#     filtered_df = df[df['actor1'] == 'Taliban and/or Islamic State Khorasan Province (ISKP)'].copy()
#
#     # Apply text preprocessing to notes column
#     filtered_df['notes_processed'] = filtered_df['notes'].apply(preprocess_notes)
#
#     return filtered_df
#
#
# def prepare_data_for_model(df, model_type, use_processed_notes=True):
#     """Prepare data based on model type"""
#     notes_col = 'notes_processed' if use_processed_notes else 'notes'
#
#     if model_type == "Classical":
#         feature_cols = MANUAL_FEATURES + EMBEDDING_FEATURES
#         return df[feature_cols]
#
#     elif model_type == "BERT":
#         return df[[notes_col]]
#
#     elif model_type == "BERT-Hybrid":
#         feature_cols = MANUAL_FEATURES + [notes_col]
#         return df[feature_cols]
#
#     return df
#
#
# # ============================================================================
# # Model Loading Functions
# # ============================================================================
#
# @st.cache_resource
# def load_classical_model(model_path, preprocessor_path):
#     """Load a classical ML model and its preprocessor saved with joblib."""
#     # Load model
#     try:
#         model = joblib.load(model_path)
#     except Exception as e:
#         raise Exception(f"Could not load model from {model_path}. Error: {str(e)}")
#
#     # Load preprocessor(s)
#     try:
#         preprocessor = joblib.load(preprocessor_path)
#     except Exception as e:
#         raise Exception(f"Could not load preprocessor from {preprocessor_path}. Error: {str(e)}")
#
#     return model, preprocessor
#
#
#
# @st.cache_resource
# def load_bert_model(model_dir):
#     """Load BERT model"""
#     model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
#     tokenizer = BertTokenizer.from_pretrained(model_dir)
#     model.eval()
#     return model, tokenizer
#
#
# @st.cache_resource
# def load_bert_hybrid_model(model_dir):
#     """Load BERT-Hybrid (Feature Fusion) model"""
#     checkpoint = torch.load(f"{model_dir}/model.pt", map_location=device)
#     num_manual_features = checkpoint.get('num_manual_features', 14)
#     prediction_threshold = checkpoint.get('prediction_threshold', 0.5)
#
#     model = BERTWithManualFeatures(
#         bert_model_name='bert-base-uncased',
#         num_manual_features=num_manual_features,
#         hidden_dim=128,
#         dropout=0.3
#     )
#
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     model.eval()
#
#     tokenizer = BertTokenizer.from_pretrained(model_dir)
#
#     # Load scaler - CRITICAL for correct predictions!
#     # Model was trained with StandardScaler (see main.py lines 460-470)
#     scaler_path = f"{model_dir}/feature_scaler.pkl"
#     scaler = None
#     if pathlib.Path(scaler_path).exists():
#         try:
#             # Try normal loading first
#             try:
#                 scaler = joblib.load(scaler_path)
#                 st.success("✅ Feature scaler loaded successfully")
#             except Exception as e:
#                 st.error(f"❌ Could not load scaler from {scaler_path}")
#                 st.error(f"Error: {str(e)}")
#                 scaler = None
#             st.success("✅ Feature scaler loaded successfully")
#         except Exception as e:
#             # If that fails, try with different encoding (Python 2/3 compatibility)
#             try:
#                 with open(scaler_path, 'rb') as f:
#                     scaler = pickle.load(f, encoding='latin1')
#                 st.success("✅ Feature scaler loaded successfully (with encoding fallback)")
#             except Exception as e2:
#                 st.error(f"❌ Could not load scaler from {scaler_path}. Predictions will be INCORRECT without it!")
#                 st.error(f"Error: {str(e2)}")
#                 scaler = None
#     else:
#         st.error(f"❌ Scaler file not found: {scaler_path}")
#         st.error("The model was trained WITH feature scaling. Predictions will be incorrect without the scaler!")
#
#     return model, tokenizer, scaler, prediction_threshold
#
#
# # ============================================================================
# # Batch Prediction Functions
# # ============================================================================
#
# def predict_classical(model, preprocessor, X):
#     """Make predictions with classical model"""
#     X_processed = preprocessor.transform(X)
#     predictions = model.predict(X_processed)
#     probabilities = model.predict_proba(X_processed) if hasattr(model, 'predict_proba') else None
#     return predictions, probabilities
#
#
# def predict_bert(model, tokenizer, texts, batch_size=16, max_length=128):
#     """Make predictions with BERT model"""
#     dataset = TextDataset(texts.tolist(), tokenizer, max_length)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#
#     predictions = []
#     probabilities = []
#
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             logits = outputs.logits
#             probs = torch.softmax(logits, dim=1)
#             preds = torch.argmax(probs, dim=1)
#
#             predictions.extend(preds.cpu().numpy())
#             probabilities.extend(probs.cpu().numpy())
#
#     return np.array(predictions), np.array(probabilities)
#
#
# def predict_bert_hybrid(model, tokenizer, scaler, texts, manual_features,
#                         batch_size=16, max_length=320, threshold=0.5):
#     """Make predictions with BERT-Hybrid model"""
#     # CRITICAL: Apply the same scaling used during training!
#     # See main.py lines 460-470 where StandardScaler is applied
#     if scaler is not None:
#         manual_features = scaler.transform(manual_features)
#     else:
#         # WARNING: Predictions will be incorrect without scaling!
#         st.warning("⚠️ No scaler available - predictions may be incorrect!")
#
#     dataset = TextFeatureDataset(
#         texts.tolist(),
#         manual_features,
#         tokenizer,
#         max_length
#     )
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#
#     predictions = []
#     probabilities = []
#
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             manual_feat = batch['manual_features'].to(device)
#
#             logits = model(input_ids, attention_mask, manual_feat)
#             probs = torch.softmax(logits, dim=1)
#             preds = (probs[:, 1] >= threshold).long()
#
#             predictions.extend(preds.cpu().numpy())
#             probabilities.extend(probs.cpu().numpy())
#
#     return np.array(predictions), np.array(probabilities)
#
#
# # ============================================================================
# # Streamlit App
# # ============================================================================
#
# def main():
#     st.set_page_config(page_title="Taliban vs ISIS-K Classifier", layout="wide")
#
#     st.title("🎯 Taliban vs ISIS-K Attack Classifier - Batch Predictions")
#
#     st.markdown("""
#     This demo classifies all conflict events at once using three different approaches:
#     - **Classical ML**: Random Forest, XGBoost, MLP with manual features + embeddings
#     - **BERT**: Fine-tuned BERT using preprocessed event descriptions
#     - **BERT-Hybrid**: Feature fusion combining BERT with manual features
#
#     **Text Preprocessing**: Group names (Taliban, ISIS-K) and location names are removed before classification.
#     """)
#
#     # ---- Sidebar: Model Selection ----
#     with st.sidebar:
#         st.header("⚙️ Configuration")
#
#         model_type = st.selectbox(
#             "Model Type:",
#             ["Classical", "BERT", "BERT-Hybrid"],
#             help="Select which modeling approach to use"
#         )
#
#         # Load available models
#         # Map model type to directory name
#         dir_mapping = {
#             'classical': 'classical',
#             'bert': 'bert',
#             'bert-hybrid': 'feature_fusion'  # BERT-Hybrid models are in feature_fusion directory
#         }
#         model_dir = SAVED_MODELS_DIR / dir_mapping[model_type.lower()]
#
#         if model_type == "Classical":
#             if model_dir.exists():
#                 model_files = [f.name for f in model_dir.iterdir() if
#                                f.suffix == '.pkl' and 'preprocessor' not in f.name]
#                 selected_model = st.selectbox("Select Model:", model_files)
#             else:
#                 st.error(f"Model directory not found: {model_dir}")
#                 return
#         else:
#             selected_model = "default"
#
#         st.divider()
#         st.markdown(f"**Model Type:** {model_type}")
#         st.markdown(f"**Device:** {device}")
#
#     # ---- Load Data ----
#     st.header("📊 Data")
#
#     sample_csv = DEMO_DIR / "unattributed_attacks_processed.csv"
#
#     if not sample_csv.exists():
#         st.error(f"❌ Data file not found: {sample_csv}")
#         return
#
#     # Load and filter data
#     with st.spinner("Loading data..."):
#         df = load_and_filter_data(sample_csv)
#
#     st.success(f"✅ Loaded **{len(df)}** Taliban/ISIS-K attacks for classification")
#
#     # Prepare data based on model type
#     prepared_data = prepare_data_for_model(df, model_type)
#
#     # Show data info
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Samples", len(prepared_data))
#     with col2:
#         if model_type == "BERT":
#             st.metric("Features", "Text Only")
#         else:
#             st.metric("Features", len(prepared_data.columns))
#     with col3:
#         st.metric("Model Type", model_type)
#
#     # Show data preview
#     with st.expander("👁️ View Data Preview (first 10 rows)"):
#         if model_type == "BERT":
#             preview_df = pd.DataFrame({
#                 'event_id': df['event_id_cnty'].head(10).values,
#                 'date': df['event_date'].head(10).values,
#                 'location': df['location'].head(10).values,
#                 'notes_preprocessed': prepared_data['notes_processed'].head(10).values
#             })
#             st.dataframe(preview_df, use_container_width=True, height=300)
#         else:
#             st.dataframe(prepared_data.head(10), use_container_width=True, height=300)
#
#     # ---- Load Model and Make Predictions ----
#     st.header("🤖 Batch Predictions")
#
#     if st.button("🚀 Run Predictions on All Events", type="primary", use_container_width=True):
#         try:
#             with st.spinner(f"Loading {model_type} model..."):
#                 if model_type == "Classical":
#                     model_path = model_dir / selected_model
#                     preprocessor_path = model_dir / "preprocessors.pkl"
#                     model, preprocessor = load_classical_model(model_path, preprocessor_path)
#
#                 elif model_type == "BERT":
#                     model, tokenizer = load_bert_model(model_dir)
#
#                 elif model_type == "BERT-Hybrid":
#                     model, tokenizer, scaler, threshold = load_bert_hybrid_model(model_dir)
#
#             st.success("✅ Model loaded successfully!")
#
#             # Make predictions
#             with st.spinner(f"Making predictions on {len(df)} events..."):
#                 # Check files exist first
#                 if model_type == "Classical":
#                     if not (model_dir / selected_model).exists():
#                         st.error(f"Model file not found: {model_dir / selected_model}")
#                         st.stop()
#                     if not (model_dir / "preprocessors.pkl").exists():
#                         st.error(f"Preprocessor file not found: {model_dir / 'preprocessors.pkl'}")
#                         st.stop()
#
#                 elif model_type == "BERT-Hybrid":
#                     if not (model_dir / "model.pt").exists():
#                         st.error(f"Model file not found: {model_dir / 'model.pt'}")
#                         st.info(f"Looking in directory: {model_dir}")
#                         st.info(
#                             f"Files available: {list(model_dir.iterdir()) if model_dir.exists() else 'Directory does not exist'}")
#                         st.stop()
#
#                 # Now make predictions
#                 if model_type == "Classical":
#                     X = prepared_data.values
#                     predictions, probabilities = predict_classical(model, preprocessor, X)
#
#                 elif model_type == "BERT":
#                     texts = prepared_data['notes_processed']
#                     predictions, probabilities = predict_bert(model, tokenizer, texts)
#
#                 elif model_type == "BERT-Hybrid":
#                     texts = prepared_data['notes_processed']
#                     manual_feats = prepared_data[MANUAL_FEATURES].values
#                     predictions, probabilities = predict_bert_hybrid(
#                         model, tokenizer, scaler, texts, manual_feats, threshold=threshold
#                     )
#
#             st.success("✅ Predictions complete!")
#
#             # Display results
#             st.subheader("📈 Results Summary")
#
#             # Class distribution
#             pred_counts = pd.Series(predictions).value_counts()
#             col1, col2 = st.columns(2)
#
#             with col1:
#                 st.metric("Predicted Taliban (Class 0)",
#                           int(pred_counts.get(0, 0)),
#                           delta=f"{pred_counts.get(0, 0) / len(predictions) * 100:.1f}%")
#
#             with col2:
#                 st.metric("Predicted ISIS-K (Class 1)",
#                           int(pred_counts.get(1, 0)),
#                           delta=f"{pred_counts.get(1, 0) / len(predictions) * 100:.1f}%")
#
#             # Create results dataframe
#             results_df = df[['event_id_cnty', 'event_date', 'location', 'fatalities', 'notes']].copy()
#             results_df['prediction'] = predictions
#             results_df['prediction_label'] = results_df['prediction'].map({0: 'Taliban', 1: 'ISIS-K'})
#
#             if probabilities is not None:
#                 results_df['prob_taliban'] = probabilities[:, 0]
#                 results_df['prob_isis_k'] = probabilities[:, 1]
#                 results_df['confidence'] = np.max(probabilities, axis=1)
#
#             # Display results table
#             st.subheader("📋 Detailed Results")
#
#             display_cols = ['event_id_cnty', 'event_date', 'location', 'fatalities',
#                             'prediction_label', 'confidence', 'prob_taliban',
#                             'prob_isis_k'] if probabilities is not None else \
#                 ['event_id_cnty', 'event_date', 'location', 'fatalities', 'prediction_label']
#
#             st.dataframe(
#                 results_df[display_cols],
#                 use_container_width=True,
#                 height=400
#             )
#
#             # Show sample predictions with notes
#             with st.expander("📝 View Sample Predictions with Event Descriptions"):
#                 sample_size = min(10, len(results_df))
#                 sample_df = results_df[['event_id_cnty', 'event_date', 'location', 'notes',
#                                         'prediction_label', 'confidence']].head(sample_size)
#
#                 for idx, row in sample_df.iterrows():
#                     st.markdown(f"**{row['event_id_cnty']}** - {row['event_date']} - {row['location']}")
#                     st.write(f"*{row['notes'][:200]}...*")
#                     st.markdown(
#                         f"**Prediction:** {row['prediction_label']} ({row['confidence'] * 100:.1f}% confidence)")
#                     st.divider()
#
#             # Download results
#             csv = results_df.to_csv(index=False)
#             st.download_button(
#                 label="📥 Download Full Results as CSV",
#                 data=csv,
#                 file_name=f"predictions_{model_type.lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                 mime="text/csv",
#                 use_container_width=True
#             )
#
#         except Exception as e:
#             st.error(f"❌ Error during prediction: {str(e)}")
#             st.exception(e)
#
#
# if __name__ == "__main__":
#     main()
#
#


# import streamlit as st
# import pandas as pd
# import numpy as np
# import pathlib
# import pickle
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, BertModel
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# import warnings
# import re
# import joblib
#
# warnings.filterwarnings('ignore')
#
# PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
# SAVED_MODELS_DIR = PROJECT_ROOT / "src" / "saved_models"
# DEMO_DIR = PROJECT_ROOT / "demo"
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Define feature sets
# MANUAL_FEATURES = [
#     'civilian_targeting',
#     'fatalities',
#     'violence_against_women',
#     'sub_event_type_Abduction/forced disappearance',
#     'sub_event_type_Air/drone strike',
#     'sub_event_type_Armed clash',
#     'sub_event_type_Attack',
#     'sub_event_type_Government regains territory',
#     'sub_event_type_Grenade',
#     'sub_event_type_Non-state actor overtakes territory',
#     'sub_event_type_Remote explosive/landmine/IED',
#     'sub_event_type_Sexual violence',
#     'sub_event_type_Shelling/artillery/missile attack',
#     'sub_event_type_Suicide bomb'
# ]
#
# EMBEDDING_FEATURES = [f'notes_emb_{i}' for i in range(384)]
#
#
# # ============================================================================
# # Text Preprocessing Functions
# # ============================================================================
#
# def mask_group_names(text):
#     """Remove all variants of Taliban and ISIS-K group names"""
#     if not isinstance(text, str):
#         return text
#
#     taliban_variants = [
#         r'\bTaliban\b', r'\bTaleban\b', r'\bTaliban-e\b', r'\bT-Ban\b',
#         r'\bTban\b', r'\bTTP\b', r'\bTehrik-i-Taliban\b', r'\bTehreek-e-Taliban\b',
#     ]
#
#     isis_variants = [
#         r'\bISIS-K\b', r'\bISIS-KP\b', r'\bISIS Khorasan\b', r'\bISIS-Khorasan\b',
#         r'\bISIL-K\b', r'\bISIL Khorasan\b', r'\bIS-K\b', r'\bIS-KP\b',
#         r'\bIslamic State Khorasan\b', r'\bIslamic State in Khorasan\b',
#         r'\bIslamic State of Khorasan\b', r'\bDaesh\b', r"\bDa\'esh\b",
#         r'\bISIS\b', r'\bISIL\b', r'\bIslamic State\b', r'\bIS\b'
#     ]
#
#     all_variants = taliban_variants + isis_variants
#     masked_text = text
#
#     for pattern in all_variants:
#         masked_text = re.sub(pattern, '', masked_text, flags=re.IGNORECASE)
#
#     masked_text = re.sub(r'\s+', ' ', masked_text).strip()
#     return masked_text
#
#
# def mask_location_names(text):
#     """Remove all location names (provinces and districts) from text"""
#     if not isinstance(text, str):
#         return text
#
#     provinces = ['Wardak', 'Shewa', 'Farah', 'Helmand', 'Kandahar', 'Nimruz', 'Paktia', 'Paktika',
#                  'Parwan', 'Logar', 'Kunar', 'Ghazni', 'Laghman', 'Herat', 'Kabul', 'Sar-e Pol',
#                  'Nangarhar', 'Nangahar', 'Khost', 'Zabul', 'Badakhshan', 'Faryab', 'Kunduz', 'Badghis',
#                  'Balkh', 'Kapisa', 'Baghlan', 'Samangan', 'Urozgan', 'Jowzjan', 'Daykundi',
#                  'Nuristan', 'Takhar', 'Ghor', 'Bamyan', 'Panjshir', 'Khatlon']
#
#     districts = ['Sayyid Abad', 'Bala Buluk', 'Lashkargah', 'Nahr-i-Saraj', 'Maruf', 'Khashrod',
#                  'Zurmat', 'Sar Rawza', 'Bagram', 'Khak-i-Safed', 'Baraki Barak', 'Sangin',
#                  'Sar Kani', 'Waghaz', 'Charkh', 'Dawlat Shah', 'Kushk-i-Kuhna', 'Surubi',
#                  'Herat', 'Sancharak', 'Shinwar', 'Ghazi Abad', 'Andar', 'Nawa-i-Barikzayi',
#                  'Nad Ali', 'Mata Khan', 'Puli Alam', 'Khost', 'Shemel Zayi', 'Gelan', 'Maiwand',
#                  'Ghazni', 'Bati Kot', 'Faiz Abad', 'Gardez', 'Almar', 'Dangam', 'Rodat', 'Nesh',
#                  'Spin Boldak', 'Mohammad Agha', 'Farah', 'Khan Abad', 'Shindand', 'Pushtrud',
#                  'Jalalabad', 'Mazar-e-Sharif', 'Kandahar', 'Kunduz', 'Kabul']
#
#     all_locations = provinces + districts
#     all_locations.sort(key=len, reverse=True)
#
#     masked_text = text
#     for location in all_locations:
#         pattern = r'\b' + re.escape(location) + r'\b'
#         masked_text = re.sub(pattern, '', masked_text, flags=re.IGNORECASE)
#
#     masked_text = re.sub(r'\s+', ' ', masked_text).strip()
#     return masked_text
#
#
# def preprocess_notes(text):
#     """Apply all preprocessing to notes text"""
#     text = mask_group_names(text)
#     text = mask_location_names(text)
#     return text
#
#
# # ============================================================================
# # Model Architecture Definitions
# # ============================================================================
#
# class TextDataset(Dataset):
#     """Dataset for BERT text classification"""
#
#     def __init__(self, texts, tokenizer, max_length=128):
#         self.texts = texts
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten()
#         }
#
#
# class TextFeatureDataset(Dataset):
#     """Dataset for BERT-Hybrid (text + manual features)"""
#
#     def __init__(self, texts, manual_features, tokenizer, max_length=320):
#         self.texts = texts
#         self.manual_features = manual_features
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         manual_feat = self.manual_features[idx]
#
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'manual_features': torch.tensor(manual_feat, dtype=torch.float)
#         }
#
#
# class BERTWithManualFeatures(nn.Module):
#     """BERT model with feature fusion for BERT-Hybrid"""
#
#     def __init__(self, bert_model_name='bert-base-uncased',
#                  num_manual_features=14,
#                  hidden_dim=128,
#                  dropout=0.3):
#         super(BERTWithManualFeatures, self).__init__()
#
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.feature_layer = nn.Linear(num_manual_features, hidden_dim)
#
#         bert_hidden_size = self.bert.config.hidden_size
#         self.fusion = nn.Sequential(
#             nn.Linear(bert_hidden_size + hidden_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#         )
#
#         self.classifier = nn.Linear(128, 2)
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()
#
#     def forward(self, input_ids, attention_mask, manual_features):
#         bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         cls_embedding = bert_output.last_hidden_state[:, 0, :]
#
#         feature_embedding = self.relu(self.feature_layer(manual_features))
#         feature_embedding = self.dropout(feature_embedding)
#
#         combined = torch.cat([cls_embedding, feature_embedding], dim=1)
#         x = self.fusion(combined)
#         logits = self.classifier(x)
#         return logits
#
#
# # ============================================================================
# # Data Processing Functions
# # ============================================================================
#
# def load_and_filter_data(csv_path):
#     """Load CSV and filter to Taliban/ISIS-K attacks only"""
#     df = pd.read_csv(csv_path)
#     filtered_df = df[df['actor1'] == 'Taliban and/or Islamic State Khorasan Province (ISKP)'].copy()
#
#     # Apply text preprocessing to notes column
#     filtered_df['notes_processed'] = filtered_df['notes'].apply(preprocess_notes)
#
#     return filtered_df
#
#
# def prepare_data_for_model(df, model_type, use_processed_notes=True):
#     """Prepare data based on model type"""
#     notes_col = 'notes_processed' if use_processed_notes else 'notes'
#
#     if model_type == "Classical":
#         feature_cols = MANUAL_FEATURES + EMBEDDING_FEATURES
#         return df[feature_cols]
#
#     elif model_type == "BERT":
#         return df[[notes_col]]
#
#     elif model_type == "BERT-Hybrid":
#         feature_cols = MANUAL_FEATURES + [notes_col]
#         return df[feature_cols]
#
#     return df
#
#
# # ============================================================================
# # Model Loading Functions
# # ============================================================================
#
# @st.cache_resource
# def load_classical_model(model_path, preprocessor_path):
#     """Load a classical ML model and its preprocessor saved with joblib."""
#     # Load model
#     try:
#         model = joblib.load(model_path)
#     except Exception as e:
#         raise Exception(f"Could not load model from {model_path}. Error: {str(e)}")
#
#     # Load preprocessor(s)
#     try:
#         preprocessor = joblib.load(preprocessor_path)
#     except Exception as e:
#         raise Exception(f"Could not load preprocessor from {preprocessor_path}. Error: {str(e)}")
#
#     return model, preprocessor
#
#
#
# @st.cache_resource
# def load_bert_model(model_dir):
#     """Load BERT model"""
#     model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
#     tokenizer = BertTokenizer.from_pretrained(model_dir)
#     model.eval()
#     return model, tokenizer
#
#
# @st.cache_resource
# def load_bert_hybrid_model(model_dir):
#     """Load BERT-Hybrid (Feature Fusion) model"""
#     checkpoint = torch.load(f"{model_dir}/model.pt", map_location=device)
#     num_manual_features = checkpoint.get('num_manual_features', 14)
#     prediction_threshold = checkpoint.get('prediction_threshold', 0.5)
#
#     model = BERTWithManualFeatures(
#         bert_model_name='bert-base-uncased',
#         num_manual_features=num_manual_features,
#         hidden_dim=128,
#         dropout=0.3
#     )
#
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     model.eval()
#
#     tokenizer = BertTokenizer.from_pretrained(model_dir)
#
#     # Load scaler - CRITICAL for correct predictions!
#     # Model was trained with StandardScaler (see main.py lines 460-470)
#     scaler_path = f"{model_dir}/feature_scaler.pkl"
#     scaler = None
#     if pathlib.Path(scaler_path).exists():
#         try:
#             # Try normal loading first
#             try:
#                 scaler = joblib.load(scaler_path)
#                 st.success("✅ Feature scaler loaded successfully")
#             except Exception as e:
#                 st.error(f"❌ Could not load scaler from {scaler_path}")
#                 st.error(f"Error: {str(e)}")
#                 scaler = None
#             st.success("✅ Feature scaler loaded successfully")
#         except Exception as e:
#             # If that fails, try with different encoding (Python 2/3 compatibility)
#             try:
#                 with open(scaler_path, 'rb') as f:
#                     scaler = pickle.load(f, encoding='latin1')
#                 st.success("✅ Feature scaler loaded successfully (with encoding fallback)")
#             except Exception as e2:
#                 st.error(f"❌ Could not load scaler from {scaler_path}. Predictions will be INCORRECT without it!")
#                 st.error(f"Error: {str(e2)}")
#                 scaler = None
#     else:
#         st.error(f"❌ Scaler file not found: {scaler_path}")
#         st.error("The model was trained WITH feature scaling. Predictions will be incorrect without the scaler!")
#
#     return model, tokenizer, scaler, prediction_threshold
#
#
# # ============================================================================
# # Batch Prediction Functions
# # ============================================================================
#
# def predict_classical(model, preprocessor, X):
#     """Make predictions with classical model"""
#     X_processed = preprocessor.transform(X)
#     predictions = model.predict(X_processed)
#     probabilities = model.predict_proba(X_processed) if hasattr(model, 'predict_proba') else None
#     return predictions, probabilities
#
#
# def predict_bert(model, tokenizer, texts, batch_size=16, max_length=128):
#     """Make predictions with BERT model"""
#     dataset = TextDataset(texts.tolist(), tokenizer, max_length)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#
#     predictions = []
#     probabilities = []
#
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             logits = outputs.logits
#             probs = torch.softmax(logits, dim=1)
#             preds = torch.argmax(probs, dim=1)
#
#             predictions.extend(preds.cpu().numpy())
#             probabilities.extend(probs.cpu().numpy())
#
#     return np.array(predictions), np.array(probabilities)
#
#
# def predict_bert_hybrid(model, tokenizer, scaler, texts, manual_features,
#                         batch_size=16, max_length=320, threshold=0.5):
#     """Make predictions with BERT-Hybrid model"""
#     # CRITICAL: Apply the same scaling used during training!
#     # See main.py lines 460-470 where StandardScaler is applied
#     if scaler is not None:
#         manual_features = scaler.transform(manual_features)
#     else:
#         # WARNING: Predictions will be incorrect without scaling!
#         st.warning("⚠️ No scaler available - predictions may be incorrect!")
#
#     dataset = TextFeatureDataset(
#         texts.tolist(),
#         manual_features,
#         tokenizer,
#         max_length
#     )
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#
#     predictions = []
#     probabilities = []
#
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             manual_feat = batch['manual_features'].to(device)
#
#             logits = model(input_ids, attention_mask, manual_feat)
#             probs = torch.softmax(logits, dim=1)
#             preds = (probs[:, 1] >= threshold).long()
#
#             predictions.extend(preds.cpu().numpy())
#             probabilities.extend(probs.cpu().numpy())
#
#     return np.array(predictions), np.array(probabilities)
#
#
# # ============================================================================
# # Streamlit App
# # ============================================================================
#
# def main():
#     st.set_page_config(page_title="Taliban vs ISIS-K Classifier", layout="wide")
#
#     st.title("🎯 Taliban vs ISIS-K Attack Classifier - Batch Predictions")
#
#     st.markdown("""
#     This demo classifies all conflict events at once using three different approaches:
#     - **Classical ML**: Random Forest, XGBoost, MLP with manual features + embeddings
#     - **BERT**: Fine-tuned BERT using preprocessed event descriptions
#     - **BERT-Hybrid**: Feature fusion combining BERT with manual features
#
#     **Text Preprocessing**: Group names (Taliban, ISIS-K) and location names are removed before classification.
#     """)
#
#     # ---- Sidebar: Model Selection ----
#     with st.sidebar:
#         st.header("⚙️ Configuration")
#
#         model_type = st.selectbox(
#             "Model Type:",
#             ["Classical", "BERT", "BERT-Hybrid"],
#             help="Select which modeling approach to use"
#         )
#
#         # Load available models
#         # Map model type to directory name
#         dir_mapping = {
#             'classical': 'classical',
#             'bert': 'bert',
#             'bert-hybrid': 'feature_fusion'  # BERT-Hybrid models are in feature_fusion directory
#         }
#         model_dir = SAVED_MODELS_DIR / dir_mapping[model_type.lower()]
#
#         if model_type == "Classical":
#             if model_dir.exists():
#                 model_files = [f.name for f in model_dir.iterdir() if
#                                f.suffix == '.pkl' and 'preprocessor' not in f.name]
#                 selected_model = st.selectbox("Select Model:", model_files)
#             else:
#                 st.error(f"Model directory not found: {model_dir}")
#                 return
#         else:
#             selected_model = "default"
#
#         st.divider()
#         st.markdown(f"**Model Type:** {model_type}")
#         st.markdown(f"**Device:** {device}")
#
#     # ---- Load Data ----
#     st.header("📊 Data")
#
#     sample_csv = DEMO_DIR / "unattributed_attacks_processed.csv"
#
#     if not sample_csv.exists():
#         st.error(f"❌ Data file not found: {sample_csv}")
#         return
#
#     # Load and filter data
#     with st.spinner("Loading data..."):
#         df = load_and_filter_data(sample_csv)
#
#     st.success(f"✅ Loaded **{len(df)}** Taliban/ISIS-K attacks for classification")
#
#     # Prepare data based on model type
#     prepared_data = prepare_data_for_model(df, model_type)
#
#     # Show data info
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Samples", len(prepared_data))
#     with col2:
#         if model_type == "BERT":
#             st.metric("Features", "Text Only")
#         else:
#             st.metric("Features", len(prepared_data.columns))
#     with col3:
#         st.metric("Model Type", model_type)
#
#     # Show data preview
#     with st.expander("👁️ View Data Preview (first 10 rows)"):
#         if model_type == "BERT":
#             preview_df = pd.DataFrame({
#                 'event_id': df['event_id_cnty'].head(10).values,
#                 'date': df['event_date'].head(10).values,
#                 'location': df['location'].head(10).values,
#                 'notes_preprocessed': prepared_data['notes_processed'].head(10).values
#             })
#             st.dataframe(preview_df, use_container_width=True, height=300)
#         else:
#             st.dataframe(prepared_data.head(10), use_container_width=True, height=300)
#
#     # ---- Load Model and Make Predictions ----
#     st.header("🤖 Batch Predictions")
#
#     if st.button("🚀 Run Predictions on All Events", type="primary", use_container_width=True):
#         try:
#             with st.spinner(f"Loading {model_type} model..."):
#                 if model_type == "Classical":
#                     model_path = model_dir / selected_model
#                     preprocessor_path = model_dir / "preprocessors.pkl"
#                     model, preprocessor = load_classical_model(model_path, preprocessor_path)
#
#                 elif model_type == "BERT":
#                     model, tokenizer = load_bert_model(model_dir)
#
#                 elif model_type == "BERT-Hybrid":
#                     model, tokenizer, scaler, threshold = load_bert_hybrid_model(model_dir)
#
#             st.success("✅ Model loaded successfully!")
#
#             # Make predictions
#             with st.spinner(f"Making predictions on {len(df)} events..."):
#                 # Check files exist first
#                 if model_type == "Classical":
#                     if not (model_dir / selected_model).exists():
#                         st.error(f"Model file not found: {model_dir / selected_model}")
#                         st.stop()
#                     if not (model_dir / "preprocessors.pkl").exists():
#                         st.error(f"Preprocessor file not found: {model_dir / 'preprocessors.pkl'}")
#                         st.stop()
#
#                 elif model_type == "BERT-Hybrid":
#                     if not (model_dir / "model.pt").exists():
#                         st.error(f"Model file not found: {model_dir / 'model.pt'}")
#                         st.info(f"Looking in directory: {model_dir}")
#                         st.info(
#                             f"Files available: {list(model_dir.iterdir()) if model_dir.exists() else 'Directory does not exist'}")
#                         st.stop()
#
#                 # Now make predictions
#                 if model_type == "Classical":
#                     X = prepared_data.values
#                     predictions, probabilities = predict_classical(model, preprocessor, X)
#
#                 elif model_type == "BERT":
#                     texts = prepared_data['notes_processed']
#                     predictions, probabilities = predict_bert(model, tokenizer, texts)
#
#                 elif model_type == "BERT-Hybrid":
#                     texts = prepared_data['notes_processed']
#                     manual_feats = prepared_data[MANUAL_FEATURES].values
#                     predictions, probabilities = predict_bert_hybrid(
#                         model, tokenizer, scaler, texts, manual_feats, threshold=threshold
#                     )
#
#             st.success("✅ Predictions complete!")
#
#             # Display results
#             st.subheader("📈 Results Summary")
#
#             # Class distribution
#             pred_counts = pd.Series(predictions).value_counts()
#             col1, col2 = st.columns(2)
#
#             with col1:
#                 st.metric("Predicted Taliban (Class 0)",
#                           int(pred_counts.get(0, 0)),
#                           delta=f"{pred_counts.get(0, 0) / len(predictions) * 100:.1f}%")
#
#             with col2:
#                 st.metric("Predicted ISIS-K (Class 1)",
#                           int(pred_counts.get(1, 0)),
#                           delta=f"{pred_counts.get(1, 0) / len(predictions) * 100:.1f}%")
#
#             # Create results dataframe
#             results_df = df[['event_id_cnty', 'event_date', 'location', 'fatalities', 'notes']].copy()
#             results_df['prediction'] = predictions
#             results_df['prediction_label'] = results_df['prediction'].map({0: 'Taliban', 1: 'ISIS-K'})
#
#             if probabilities is not None:
#                 results_df['prob_taliban'] = probabilities[:, 0]
#                 results_df['prob_isis_k'] = probabilities[:, 1]
#                 results_df['confidence'] = np.max(probabilities, axis=1)
#
#             # Display results table
#             st.subheader("📋 Detailed Results")
#
#             display_cols = ['event_id_cnty', 'event_date', 'location', 'fatalities',
#                             'prediction_label', 'confidence', 'prob_taliban',
#                             'prob_isis_k'] if probabilities is not None else \
#                 ['event_id_cnty', 'event_date', 'location', 'fatalities', 'prediction_label']
#
#             st.dataframe(
#                 results_df[display_cols],
#                 use_container_width=True,
#                 height=400
#             )
#
#             # Show sample predictions with notes
#             with st.expander("📝 View Sample Predictions with Event Descriptions"):
#                 sample_size = min(10, len(results_df))
#                 sample_df = results_df[['event_id_cnty', 'event_date', 'location', 'notes',
#                                         'prediction_label', 'confidence']].head(sample_size)
#
#                 for idx, row in sample_df.iterrows():
#                     st.markdown(f"**{row['event_id_cnty']}** - {row['event_date']} - {row['location']}")
#                     st.write(f"*{row['notes'][:200]}...*")
#                     st.markdown(
#                         f"**Prediction:** {row['prediction_label']} ({row['confidence'] * 100:.1f}% confidence)")
#                     st.divider()
#
#             # Download results
#             csv = results_df.to_csv(index=False)
#             st.download_button(
#                 label="📥 Download Full Results as CSV",
#                 data=csv,
#                 file_name=f"predictions_{model_type.lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                 mime="text/csv",
#                 use_container_width=True
#             )
#
#         except Exception as e:
#             st.error(f"❌ Error during prediction: {str(e)}")
#             st.exception(e)
#
#
# if __name__ == "__main__":
#     main()
#
#