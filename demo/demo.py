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

warnings.filterwarnings('ignore')

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SAVED_MODELS_DIR = PROJECT_ROOT / "src" / "saved_models"
DEMO_DIR = PROJECT_ROOT / "demo"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define feature sets based on train_test_split.py
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
    return filtered_df


def prepare_data_for_model(df, model_type):
    """
    Prepare data based on model type:
    - Classical: manual features + embeddings
    - BERT: just notes text
    - BERT-Hybrid: manual features + notes text
    """
    if model_type == "Classical":
        feature_cols = MANUAL_FEATURES + EMBEDDING_FEATURES
        return df[feature_cols]

    elif model_type == "BERT":
        return df[['notes']]

    elif model_type == "BERT-Hybrid":
        feature_cols = MANUAL_FEATURES + ['notes']
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
    # Load model checkpoint
    checkpoint = torch.load(f"{model_dir}/model.pt", map_location=device)
    num_manual_features = checkpoint.get('num_manual_features', 14)
    prediction_threshold = checkpoint.get('prediction_threshold', 0.5)

    # Create model architecture
    model = BERTWithManualFeatures(
        bert_model_name='bert-base-uncased',
        num_manual_features=num_manual_features,
        hidden_dim=128,
        dropout=0.3
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # Load feature scaler if exists
    scaler_path = f"{model_dir}/feature_scaler.pkl"
    scaler = None
    if pathlib.Path(scaler_path).exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    return model, tokenizer, scaler, prediction_threshold


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_classical(model, preprocessor, X):
    """Make predictions with classical model"""
    X_processed = preprocessor.transform(X)
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
    # Scale manual features if scaler exists
    if scaler is not None:
        manual_features = scaler.transform(manual_features)

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
# Streamlit App
# ============================================================================

def main():
    st.set_page_config(page_title="Taliban vs ISIS-K Classifier", layout="wide")

    st.title("🎯 Taliban vs ISIS-K Attack Classifier Demo")

    st.markdown("""
    This demo showcases three different approaches to classifying conflict events in Afghanistan:
    - **Classical ML**: Random Forest, XGBoost, MLP with manual features + embeddings
    - **BERT**: Fine-tuned BERT using event descriptions
    - **BERT-Hybrid**: Feature fusion combining BERT with manual features
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
    st.header("📊 Data")

    sample_csv = DEMO_DIR / "unattributed_attacks_processed.csv"

    if not sample_csv.exists():
        st.error(f"❌ Data file not found: {sample_csv}")
        return

    # Load and filter data
    with st.spinner("Loading data..."):
        df = load_and_filter_data(sample_csv)

    st.success(f"✅ Loaded **{len(df)}** Taliban/ISIS-K attacks for classification")

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
    with st.expander("👁️ View Data Preview"):
        if model_type == "BERT":
            st.dataframe(prepared_data.head(10), use_container_width=True, height=300)
        else:
            st.dataframe(prepared_data.head(10), use_container_width=True, height=300)

    # ---- Load Model and Make Predictions ----
    st.header("🤖 Model Predictions")

    if st.button("🚀 Run Predictions", type="primary"):
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

            st.success("✅ Model loaded successfully!")

            # Make predictions
            with st.spinner("Making predictions..."):
                if model_type == "Classical":
                    X = prepared_data.values
                    predictions, probabilities = predict_classical(model, preprocessor, X)

                elif model_type == "BERT":
                    texts = prepared_data['notes']
                    predictions, probabilities = predict_bert(model, tokenizer, texts)

                elif model_type == "BERT-Hybrid":
                    texts = prepared_data['notes']
                    manual_feats = prepared_data[MANUAL_FEATURES].values
                    predictions, probabilities = predict_bert_hybrid(
                        model, tokenizer, scaler, texts, manual_feats, threshold=threshold
                    )

            st.success("✅ Predictions complete!")

            # Display results
            st.subheader("📈 Results")

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
            results_df = df.copy()
            results_df['prediction'] = predictions
            results_df['prediction_label'] = results_df['prediction'].map({0: 'Taliban', 1: 'ISIS-K'})

            if probabilities is not None:
                results_df['prob_taliban'] = probabilities[:, 0]
                results_df['prob_isis_k'] = probabilities[:, 1]
                results_df['confidence'] = np.max(probabilities, axis=1)

            # Display results table
            display_cols = ['event_id_cnty', 'event_date', 'location', 'notes',
                            'prediction_label', 'confidence'] if probabilities is not None else \
                ['event_id_cnty', 'event_date', 'location', 'notes', 'prediction_label']

            st.dataframe(
                results_df[display_cols].head(20),
                use_container_width=True,
                height=400
            )

            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results as CSV",
                data=csv,
                file_name=f"predictions_{model_type.lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()