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

    # Clean up multiple spaces
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
                 'Jalalabad', 'Mazar-e-Sharif', 'Kandahar', 'Kunduz', 'Kabul']  # Shortened for brevity

    all_locations = provinces + districts
    all_locations.sort(key=len, reverse=True)

    masked_text = text
    for location in all_locations:
        pattern = r'\b' + re.escape(location) + r'\b'
        masked_text = re.sub(pattern, '', masked_text, flags=re.IGNORECASE)

    # Clean up multiple spaces
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
    filtered_df = df[df['actor1'] == 'Taliban and/or Islamic State Khorasan Province (ISKP)'].copy()

    # Apply text preprocessing to notes column
    filtered_df['notes_processed'] = filtered_df['notes'].apply(preprocess_notes)

    return filtered_df


def prepare_data_for_model(df, model_type, use_processed_notes=True):
    """
    Prepare data based on model type
    """
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
# Model Loading Functions
# ============================================================================

@st.cache_resource
def load_classical_model(model_path, preprocessor_path):
    """Load a classical ML model and its preprocessor"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)

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

    scaler_path = f"{model_dir}/feature_scaler.pkl"
    scaler = None
    if pathlib.Path(scaler_path).exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    return model, tokenizer, scaler, prediction_threshold


# ============================================================================
# Single-Row Prediction Functions
# ============================================================================

def predict_single_classical(model, preprocessor, X_single):
    """Make prediction for a single sample with classical model"""
    X_processed = preprocessor.transform(X_single.reshape(1, -1))
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0] if hasattr(model, 'predict_proba') else None
    return prediction, probability


def predict_single_bert(model, tokenizer, text, max_length=128):
    """Make prediction for a single text with BERT model"""
    encoding = tokenizer.encode_plus(
        str(text),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).cpu().numpy()[0]
        probability = probs.cpu().numpy()[0]

    return prediction, probability


def predict_single_bert_hybrid(model, tokenizer, scaler, text, manual_features,
                               max_length=320, threshold=0.5):
    """Make prediction for a single sample with BERT-Hybrid model"""
    # Scale manual features if scaler exists
    if scaler is not None:
        manual_features = scaler.transform(manual_features.reshape(1, -1))
    else:
        manual_features = manual_features.reshape(1, -1)

    encoding = tokenizer.encode_plus(
        str(text),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    manual_feat = torch.tensor(manual_features, dtype=torch.float).to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, manual_feat)
        probs = torch.softmax(logits, dim=1)
        prediction = (probs[:, 1] >= threshold).long().cpu().numpy()[0]
        probability = probs.cpu().numpy()[0]

    return prediction, probability


# ============================================================================
# Streamlit App
# ============================================================================

def main():
    st.set_page_config(page_title="Taliban vs ISIS-K Classifier", layout="wide")

    st.title("🎯 Taliban vs ISIS-K Attack Classifier Demo")

    st.markdown("""
    This demo classifies individual conflict events using three different approaches:
    - **Classical ML**: Random Forest, XGBoost, MLP with manual features + embeddings
    - **BERT**: Fine-tuned BERT using preprocessed event descriptions
    - **BERT-Hybrid**: Feature fusion combining BERT with manual features

    **Text Preprocessing**: Group names (Taliban, ISIS-K) and location names are removed before classification.
    """)

    # ---- Sidebar: Model Selection ----
    with st.sidebar:
        st.header("⚙️ Configuration")

        model_type = st.selectbox(
            "Model Type:",
            ["Classical", "BERT", "BERT-Hybrid"],
            help="Select which modeling approach to use"
        )

        # Load available models
        model_dir = SAVED_MODELS_DIR / model_type.lower().replace('-', '_')

        if model_type == "Classical":
            if model_dir.exists():
                model_files = [f.name for f in model_dir.iterdir() if
                               f.suffix == '.pkl' and 'preprocessor' not in f.name]
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
    st.header("📊 Data Selection")

    sample_csv = DEMO_DIR / "unattributed_attacks_processed.csv"

    if not sample_csv.exists():
        st.error(f"❌ Data file not found: {sample_csv}")
        return

    # Load and filter data
    with st.spinner("Loading data..."):
        df = load_and_filter_data(sample_csv)

    st.success(f"✅ Loaded **{len(df)}** Taliban/ISIS-K attacks")

    # Create display dataframe with important columns
    display_df = df[['event_id_cnty', 'event_date', 'location', 'fatalities', 'notes']].copy()
    display_df['row_number'] = range(len(display_df))

    # ---- Row Selection ----
    st.subheader("Select an Event to Classify")

    # Create options for selectbox
    options = []
    for idx, row in display_df.iterrows():
        option_text = f"Row {row['row_number']}: {row['event_date']} - {row['location']} ({row['fatalities']} deaths) - {row['notes'][:80]}..."
        options.append((idx, option_text))

    selected_idx = st.selectbox(
        "Choose an event:",
        options,
        format_func=lambda x: x[1],
        key='event_selector'
    )[0]

    # Show selected event details
    st.divider()
    st.subheader("📋 Selected Event Details")

    selected_row = df.loc[selected_idx]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Event ID", selected_row['event_id_cnty'])
        st.metric("Date", selected_row['event_date'])
    with col2:
        st.metric("Location", selected_row['location'])
        st.metric("Fatalities", int(selected_row['fatalities']))
    with col3:
        st.metric("Civilian Targeting", "Yes" if selected_row['civilian_targeting'] == 1 else "No")
        st.metric("Violence vs Women", "Yes" if selected_row.get('violence_against_women', 0) == 1 else "No")

    # Show original and processed notes
    with st.expander("📝 View Event Description"):
        st.markdown("**Original Notes:**")
        st.write(selected_row['notes'])

        st.markdown("**Preprocessed Notes (group names & locations removed):**")
        st.write(selected_row['notes_processed'])

    # ---- Make Prediction ----
    st.divider()
    st.header("🤖 Model Prediction")

    if st.button("🚀 Classify This Event", type="primary", use_container_width=True):
        try:
            # Load model
            with st.spinner(f"Loading {model_type} model..."):
                if model_type == "Classical":
                    model_path = model_dir / selected_model
                    preprocessor_path = model_dir / "preprocessors.pkl"
                    model, preprocessor = load_classical_model(model_path, preprocessor_path)

                elif model_type == "BERT":
                    model, tokenizer = load_bert_model(model_dir)

                elif model_type == "BERT-Hybrid":
                    model, tokenizer, scaler, threshold = load_bert_hybrid_model(model_dir)

            st.success("✅ Model loaded successfully!")

            # Prepare data and make prediction
            with st.spinner("Making prediction..."):
                if model_type == "Classical":
                    # Get features for this row
                    prepared = prepare_data_for_model(df, model_type)
                    X_single = prepared.iloc[selected_idx].values
                    prediction, probability = predict_single_classical(model, preprocessor, X_single)

                elif model_type == "BERT":
                    # Get preprocessed notes
                    text = selected_row['notes_processed']
                    prediction, probability = predict_single_bert(model, tokenizer, text)

                elif model_type == "BERT-Hybrid":
                    # Get preprocessed notes and manual features
                    text = selected_row['notes_processed']
                    manual_feats = selected_row[MANUAL_FEATURES].values
                    prediction, probability = predict_single_bert_hybrid(
                        model, tokenizer, scaler, text, manual_feats, threshold=threshold
                    )

            # Display results
            st.success("✅ Prediction complete!")

            st.subheader("📊 Results")

            # Prediction
            pred_label = "Taliban (Class 0)" if prediction == 0 else "ISIS-K (Class 1)"
            pred_color = "blue" if prediction == 0 else "red"

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"### Prediction")
                st.markdown(f"<h2 style='color: {pred_color};'>{pred_label}</h2>", unsafe_allow_html=True)

            with col2:
                st.markdown(f"### Confidence")
                confidence = probability[prediction] * 100
                st.markdown(f"<h2>{confidence:.1f}%</h2>", unsafe_allow_html=True)

            with col3:
                st.markdown(f"### Probabilities")
                st.write(f"Taliban: {probability[0] * 100:.1f}%")
                st.write(f"ISIS-K: {probability[1] * 100:.1f}%")

            # Probability bar chart
            prob_df = pd.DataFrame({
                'Class': ['Taliban', 'ISIS-K'],
                'Probability': [probability[0], probability[1]]
            })
            st.bar_chart(prob_df.set_index('Class'))

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()