"""
BERT Feature Fusion Model
Combines BERT text embeddings with manually engineered features
Now using improved BERT configuration from non_classical.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TextFeatureDataset(Dataset):
    """Dataset for text + manual features classification"""

    def __init__(self, texts, manual_features, labels, tokenizer, max_length=128):
        self.texts = texts
        self.manual_features = manual_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
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
            'manual_features': torch.tensor(manual_feat, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """Compute focal loss with class-dependent alpha."""
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(ce_loss, self.alpha),
                              torch.full_like(ce_loss, 1.0 - self.alpha))
        focal_loss = alpha_t * (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()


class BERTWithManualFeatures(nn.Module):
    """BERT model with feature fusion layer for combining text and manual features"""

    def __init__(self, bert_model_name='bert-base-uncased',
                 num_manual_features=10,
                 hidden_dim=128,
                 dropout=0.3):
        super(BERTWithManualFeatures, self).__init__()

        # BERT for text processing
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Linear layer for manual features
        self.feature_layer = nn.Linear(num_manual_features, hidden_dim)

        # Combined layers
        bert_hidden_size = self.bert.config.hidden_size  # 768 for base BERT
        # self.combined_layer = nn.Linear(bert_hidden_size + hidden_dim, 256)
        #
        #
        # self.classifier = nn.Linear(256, 2)
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
        # Get BERT output
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = bert_output.last_hidden_state[:, 0, :]  # [CLS] token

        # Process manual features
        feature_embedding = self.relu(self.feature_layer(manual_features))
        feature_embedding = self.dropout(feature_embedding)

        # Concatenate BERT and manual features
        # combined = torch.cat([cls_embedding, feature_embedding], dim=1)
        #
        # # Pass through combined layers
        # x = self.relu(self.combined_layer(combined))
        # x = self.dropout(x)
        #
        # # Final classification
        # logits = self.classifier(x)
        # return logits
        combined = torch.cat([cls_embedding, feature_embedding], dim=1)

        # Pass through fusion MLP
        x = self.fusion(combined)

        # Final classification
        logits = self.classifier(x)
        return logits


class BERTFeatureFusionClassifier:
    """
    BERT classifier with manual feature fusion for binary classification
    Now using improved configuration from non_classical.py
    """

    def __init__(self, model_name='bert-base-uncased',
                 max_length=128,
                 batch_size=16,
                 learning_rate=2e-5,
                 epochs=10,
                 random_state=42,
                 early_stopping_patience=2,
                 focal_loss=True,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 freeze_bert_base=True,
                 unfreeze_last_n_layers=4,
                 dropout_rate=0.3,
                 gradient_accumulation_steps=2,
                 prediction_threshold=0.5,
                 use_batch_balancing=True,
                 hidden_dim=128):
        """
        Initialize BERT Feature Fusion classifier with improved BERT configuration

        Args:
            model_name: Pretrained BERT model name
            max_length: Maximum sequence length
            batch_size: Batch size for training
            learning_rate: Learning rate
            epochs: Number of training epochs
            random_state: Random seed
            early_stopping_patience: Number of epochs to wait before early stopping
            focal_loss: Whether to use focal loss
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            freeze_bert_base: Whether to freeze BERT base layers
            unfreeze_last_n_layers: Number of top layers to unfreeze
            dropout_rate: Dropout rate
            gradient_accumulation_steps: Gradient accumulation steps
            prediction_threshold: Classification threshold
            use_batch_balancing: Whether to use WeightedRandomSampler for balanced batches
            hidden_dim: Hidden dimension for manual feature layer
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.early_stopping_patience = early_stopping_patience
        self.use_focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.freeze_bert_base = freeze_bert_base
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.dropout_rate = dropout_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.prediction_threshold = prediction_threshold
        self.use_batch_balancing = use_batch_balancing
        self.hidden_dim = hidden_dim

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        self.training_stats = []
        self.num_manual_features = None

    def _create_model(self, num_manual_features):
        """Create BERT model with feature fusion"""
        model = BERTWithManualFeatures(
            bert_model_name=self.model_name,
            num_manual_features=num_manual_features,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout_rate
        )

        # Apply freezing strategy (same as non_classical.py)
        if self.freeze_bert_base:
            # Freeze all BERT parameters
            for param in model.bert.parameters():
                param.requires_grad = False

            # Unfreeze last N layers
            if self.unfreeze_last_n_layers > 0:
                for layer in model.bert.encoder.layer[-self.unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

            # Keep feature layers and classifier trainable
            for param in model.feature_layer.parameters():
                param.requires_grad = True
            for param in model.combined_layer.parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True

        return model.to(device)

    def fit(self, X_text_train, X_features_train, y_train,
            X_text_val, X_features_val, y_val):
        """Train the model with improved BERT training loop"""
        print("\n" + "=" * 80)
        print("TRAINING BERT FEATURE FUSION MODEL")
        print("=" * 80)

        # Convert to appropriate formats
        X_text_train_list = X_text_train.tolist() if hasattr(X_text_train, 'tolist') else X_text_train
        y_train_list = y_train.tolist() if hasattr(y_train, 'tolist') else y_train
        y_train_array = np.array(y_train_list)

        # Convert manual features to numpy array
        if isinstance(X_features_train, pd.DataFrame):
            X_features_train = X_features_train.values
        X_features_train = np.array(X_features_train, dtype=np.float32)

        # Store number of features
        self.num_manual_features = X_features_train.shape[1]
        print(f"Number of manual features: {self.num_manual_features}")

        # Create datasets
        train_dataset = TextFeatureDataset(
            X_text_train_list,
            X_features_train,
            y_train_list,
            self.tokenizer,
            self.max_length
        )

        # Prepare validation data
        if isinstance(X_features_val, pd.DataFrame):
            X_features_val = X_features_val.values
        X_features_val = np.array(X_features_val, dtype=np.float32)

        val_dataset = TextFeatureDataset(
            X_text_val.tolist() if hasattr(X_text_val, 'tolist') else X_text_val,
            X_features_val,
            y_val.tolist() if hasattr(y_val, 'tolist') else y_val,
            self.tokenizer,
            self.max_length
        )

        # Create data loaders with batch balancing (same as non_classical.py)
        if self.use_batch_balancing:
            class_counts = np.bincount(y_train_array)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[y_train_array]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            print(f"\nBatch Balancing Enabled:")
            print(f"  Class 0 count: {class_counts[0]}")
            print(f"  Class 1 count: {class_counts[1]}")
            print(f"  Class 0 weight: {class_weights[0]:.4f}")
            print(f"  Class 1 weight: {class_weights[1]:.4f}")
            print(f"  Minority class will be oversampled ~{class_counts[0] / class_counts[1]:.1f}x per epoch")

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Create model
        self.model = self._create_model(self.num_manual_features)

        # Loss function
        if self.use_focal_loss:
            criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss()

        # Optimizer and scheduler (same as non_classical.py)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )

        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Training loop with early stopping
        best_val_metric = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print('=' * 80)

            # Training phase
            self.model.train()
            train_loss = 0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc="Training")):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                manual_features = batch['manual_features'].to(device)
                labels = batch['label'].to(device)

                # Forward pass
                logits = self.model(input_ids, attention_mask, manual_features)
                loss = criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * self.gradient_accumulation_steps

            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    manual_features = batch['manual_features'].to(device)
                    labels = batch['label'].to(device)

                    logits = self.model(input_ids, attention_mask, manual_features)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)

            # Calculate metrics
            report = classification_report(val_labels, val_preds, output_dict=True, zero_division=0)
            val_f1 = report['macro avg']['f1-score']
            val_minority_recall = report['1']['recall']

            # Store stats
            self.training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_f1': val_f1,
                'val_minority_recall': val_minority_recall
            })

            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Val F1: {val_f1:.4f}")
            print(f"Val Minority Recall: {val_minority_recall:.4f}")

            # Early stopping check (monitoring minority recall like non_classical.py)
            current_metric = val_minority_recall
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(f"New best model (minority recall: {best_val_metric:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement (patience: {patience_counter}/{self.early_stopping_patience})")

                if patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nLoaded best model (minority recall: {best_val_metric:.4f})")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80 + "\n")

        return self

    def predict(self, X_text_test, X_features_test):
        """Make predictions on test data"""
        self.model.eval()

        # Prepare features
        if isinstance(X_features_test, pd.DataFrame):
            X_features_test = X_features_test.values
        X_features_test = np.array(X_features_test, dtype=np.float32)

        dataset = TextFeatureDataset(
            X_text_test.tolist() if hasattr(X_text_test, 'tolist') else X_text_test,
            X_features_test,
            [0] * len(X_text_test),
            self.tokenizer,
            self.max_length
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                manual_features = batch['manual_features'].to(device)

                logits = self.model(input_ids, attention_mask, manual_features)
                probs = torch.softmax(logits, dim=1)
                preds = (probs[:, 1] >= self.prediction_threshold).long()
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X_text_test, X_features_test):
        """Get prediction probabilities"""
        self.model.eval()

        # Prepare features
        if isinstance(X_features_test, pd.DataFrame):
            X_features_test = X_features_test.values
        X_features_test = np.array(X_features_test, dtype=np.float32)

        dataset = TextFeatureDataset(
            X_text_test.tolist() if hasattr(X_text_test, 'tolist') else X_text_test,
            X_features_test,
            [0] * len(X_text_test),
            self.tokenizer,
            self.max_length
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        probabilities = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Getting probabilities"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                manual_features = batch['manual_features'].to(device)

                logits = self.model(input_ids, attention_mask, manual_features)
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def find_optimal_threshold(self, X_text_val, X_features_val, y_val):
        """Find optimal classification threshold on validation set"""
        print("\nFinding optimal classification threshold...")

        y_proba = self.predict_proba(X_text_val, X_features_val)
        y_proba_class1 = y_proba[:, 1]

        best_f1 = 0
        best_threshold = 0.5

        for threshold in np.arange(0.3, 0.8, 0.05):
            y_pred = (y_proba_class1 >= threshold).astype(int)
            report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            f1 = report['macro avg']['f1-score']

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.prediction_threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

        return best_threshold

    def evaluate(self, X_text_test, X_features_test, y_test):
        """Evaluate model on test set"""
        print("\n" + "=" * 80)
        print("EVALUATING BERT FEATURE FUSION MODEL")
        print("=" * 80)

        y_pred = self.predict(X_text_test, X_features_test)
        y_proba = self.predict_proba(X_text_test, X_features_test)

        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results = {
            'model': 'bert_feature_fusion',
            'test_f1_macro': report['macro avg']['f1-score'],
            'test_accuracy': report['accuracy'],
            'test_precision': report['macro avg']['precision'],
            'test_recall': report['macro avg']['recall'],
            'minority_recall': report['1']['recall'],
            'minority_precision': report['1']['precision'],
            'minority_f1': report['1']['f1-score']
        }

        print("\nTest Results:")
        print(f"F1 Score (Macro): {results['test_f1_macro']:.4f}")
        print(f"Accuracy: {results['test_accuracy']:.4f}")
        print(f"Precision: {results['test_precision']:.4f}")
        print(f"Recall: {results['test_recall']:.4f}")
        print(f"\nMinority Class (Class 1) Performance:")
        print(f"Recall: {results['minority_recall']:.4f}")
        print(f"Precision: {results['minority_precision']:.4f}")
        print(f"F1: {results['minority_f1']:.4f}")

        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("=" * 80 + "\n")

        return results, y_pred, y_proba

    def get_training_stats(self):
        """Return training statistics as DataFrame"""
        return pd.DataFrame(self.training_stats)

    def save_model(self, path):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_manual_features': self.num_manual_features,
            'prediction_threshold': self.prediction_threshold
        }, f"{path}/model.pt")

        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model from disk"""
        checkpoint = torch.load(f"{path}/model.pt")

        self.num_manual_features = checkpoint['num_manual_features']
        self.prediction_threshold = checkpoint['prediction_threshold']

        self.model = self._create_model(self.num_manual_features)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    print("Feature Fusion Module Loaded Successfully!")
    print("Now using improved BERT configuration from non_classical.py")


# """
# BERT Feature Fusion Model
# Combines BERT text embeddings with manually engineered features
# """
#
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from transformers import BertModel, BertTokenizer
# from torch.optim import AdamW
# from transformers import get_linear_schedule_with_warmup
# from sklearn.metrics import classification_report
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
#
#
# class TextFeatureDataset(Dataset):
#     """Dataset for text + manual features classification"""
#
#     def __init__(self, texts, manual_features, labels, tokenizer, max_length=128):
#         """
#         Args:
#             texts: List or Series of text strings
#             manual_features: numpy array or DataFrame of manual features
#             labels: List or Series of labels
#             tokenizer: BERT tokenizer
#             max_length: Maximum sequence length for BERT
#         """
#         self.texts = texts
#         self.manual_features = manual_features
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = self.labels[idx]
#         manual_feat = self.manual_features[idx]
#
#         # Tokenize text
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
#             'manual_features': torch.tensor(manual_feat, dtype=torch.float),
#             'label': torch.tensor(label, dtype=torch.long)
#         }
#
#
# class BERTWithManualFeatures(nn.Module):
#     """
#     BERT model with feature fusion layer for combining text and manual features
#     """
#
#     def __init__(self, bert_model_name='bert-base-uncased',
#                  num_manual_features=10,
#                  hidden_dim=128,
#
#                  dropout=0.3):
#         """
#         Args:
#             bert_model_name: Pretrained BERT model name
#             num_manual_features: Number of manual features to integrate
#             hidden_dim: Hidden dimension for manual feature layer
#             dropout: Dropout rate
#         """
#         super(BERTWithManualFeatures, self).__init__()
#
#         # BERT for text processing
#         self.bert = BertModel.from_pretrained(bert_model_name)
#
#         # Linear layer for manual features
#         self.feature_layer = nn.Linear(num_manual_features, hidden_dim)
#
#         # Combined layers
#         bert_hidden_size = self.bert.config.hidden_size  # 768 for base BERT
#         self.combined_layer = nn.Linear(bert_hidden_size + hidden_dim, 256)
#
#         # Final classification layer
#         self.classifier = nn.Linear(256, 2)
#
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()
#
#     def forward(self, input_ids, attention_mask, manual_features):
#         """
#         Forward pass
#
#         Args:
#             input_ids: BERT input token ids
#             attention_mask: BERT attention mask
#             manual_features: Manual features tensor
#
#         Returns:
#             logits: Classification logits
#         """
#         # Get BERT output
#         bert_output = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         cls_embedding = bert_output.last_hidden_state[:, 0, :]  # [CLS] token
#
#         # Process manual features
#         feature_embedding = self.relu(self.feature_layer(manual_features))
#         feature_embedding = self.dropout(feature_embedding)
#
#         # Concatenate BERT and manual features
#         combined = torch.cat([cls_embedding, feature_embedding], dim=1)
#
#         # Pass through combined layers
#         x = self.relu(self.combined_layer(combined))
#         x = self.dropout(x)
#
#         # Final classification
#         logits = self.classifier(x)
#         return logits
#
#
# class BERTFeatureFusionClassifier:
#     """
#     BERT classifier with manual feature fusion for binary classification
#     """
#
#     def __init__(self, model_name='bert-base-uncased',
#                  max_length=128,
#                  batch_size=16,
#                  learning_rate=2e-5,
#                  epochs=3,
#                  random_state=42,
#                  hidden_dim=128,
#                  dropout=0.3,
#                  use_class_weights=True,
#                  use_balanced_sampler=True,
#                  focal_loss=False,
#                  focal_alpha=0.25,
#                  focal_gamma=2.0,
#                  prediction_threshold=0.5):
#         """
#         Initialize BERT Feature Fusion classifier
#
#         Args:
#             model_name: Pretrained BERT model name
#             max_length: Maximum sequence length
#             batch_size: Batch size for training
#             learning_rate: Learning rate
#             epochs: Number of training epochs
#             random_state: Random seed
#             hidden_dim: Hidden dimension for manual feature layer
#             dropout: Dropout rate
#             use_class_weights: Whether to use class weights in loss
#             use_balanced_sampler: Whether to use balanced sampling
#             focal_loss: Whether to use focal loss
#             focal_alpha: Focal loss alpha parameter
#             focal_gamma: Focal loss gamma parameter
#             prediction_threshold: Classification threshold
#         """
#         self.model_name = model_name
#         self.max_length = max_length
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.random_state = random_state
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         self.use_class_weights = use_class_weights
#         self.use_balanced_sampler = use_balanced_sampler
#         self.focal_loss = focal_loss
#         self.focal_alpha = focal_alpha
#         self.focal_gamma = focal_gamma
#         self.prediction_threshold = prediction_threshold
#
#         # Set random seeds
#         torch.manual_seed(random_state)
#         np.random.seed(random_state)
#
#         # Initialize tokenizer
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = None
#         self.training_stats = []
#         self.class_weights = None
#         self.num_manual_features = None
#
#     def _calculate_class_weights(self, y_train):
#         """Calculate balanced class weights using effective number of samples"""
#         unique, counts = np.unique(y_train, return_counts=True)
#
#         # Effective number of samples (moderate weighting)
#         beta = 0.9999
#         effective_num = 1.0 - np.power(beta, counts)
#         weights_effective = (1.0 - beta) / effective_num
#         weights_effective = weights_effective / weights_effective.sum() * len(unique)
#
#         class_weights = torch.FloatTensor(weights_effective).to(device)
#
#         print(f"\nClass distribution: {dict(zip(unique, counts))}")
#         print(f"Class weights: {dict(zip(unique, weights_effective))}")
#
#         return class_weights
#
#     def _create_model(self, num_manual_features):
#         """Create BERT model with feature fusion"""
#         model = BERTWithManualFeatures(
#             bert_model_name=self.model_name,
#             num_manual_features=num_manual_features,
#             hidden_dim=self.hidden_dim,
#             dropout=self.dropout
#         )
#         return model.to(device)
#
#     def _create_balanced_sampler(self, y_train):
#         """Create a weighted sampler for balanced batches"""
#         class_counts = np.bincount(y_train)
#         class_weights = 1. / class_counts
#         sample_weights = class_weights[y_train]
#
#         sampler = WeightedRandomSampler(
#             weights=sample_weights,
#             num_samples=len(sample_weights),
#             replacement=True
#         )
#         return sampler
#
#     def _focal_loss(self, logits, labels, alpha=0.25, gamma=2.0):
#         """Focal Loss for handling class imbalance"""
#         ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = alpha * (1 - pt) ** gamma * ce_loss
#         return focal_loss.mean()
#
#     def fit(self, X_text_train, X_features_train, y_train,
#             X_text_val=None, X_features_val=None, y_val=None):
#         """
#         Train the BERT Feature Fusion model
#
#         Args:
#             X_text_train: Training texts (list or Series)
#             X_features_train: Training manual features (numpy array or DataFrame)
#             y_train: Training labels (list or Series)
#             X_text_val: Validation texts (optional)
#             X_features_val: Validation manual features (optional)
#             y_val: Validation labels (optional)
#         """
#         print("\n" + "=" * 80)
#         print("TRAINING BERT FEATURE FUSION CLASSIFIER")
#         print("=" * 80)
#
#         # Convert to appropriate formats
#         X_text_train_list = X_text_train.tolist() if hasattr(X_text_train, 'tolist') else X_text_train
#         y_train_list = y_train.tolist() if hasattr(y_train, 'tolist') else y_train
#         y_train_array = np.array(y_train_list)
#
#         # Convert manual features to numpy array
#         if isinstance(X_features_train, pd.DataFrame):
#             X_features_train = X_features_train.values
#         X_features_train = np.array(X_features_train, dtype=np.float32)
#
#         # Store number of features for model creation
#         self.num_manual_features = X_features_train.shape[1]
#         print(f"Number of manual features: {self.num_manual_features}")
#
#         # Calculate class weights
#         if self.use_class_weights:
#             self.class_weights = self._calculate_class_weights(y_train_array)
#
#         # Create model
#         self.model = self._create_model(self.num_manual_features)
#
#         # Create training dataset
#         train_dataset = TextFeatureDataset(
#             X_text_train_list,
#             X_features_train,
#             y_train_list,
#             self.tokenizer,
#             self.max_length
#         )
#
#         # Create sampler if using balanced sampling
#         sampler = None
#         shuffle = True
#         if self.use_balanced_sampler:
#             sampler = self._create_balanced_sampler(y_train_array)
#             shuffle = False
#             print("Using balanced batch sampler")
#
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=self.batch_size,
#             shuffle=shuffle,
#             sampler=sampler
#         )
#
#         # Create validation loader if provided
#         val_loader = None
#         if X_text_val is not None and X_features_val is not None and y_val is not None:
#             # Convert validation features
#             if isinstance(X_features_val, pd.DataFrame):
#                 X_features_val = X_features_val.values
#             X_features_val = np.array(X_features_val, dtype=np.float32)
#
#             val_dataset = TextFeatureDataset(
#                 X_text_val.tolist() if hasattr(X_text_val, 'tolist') else X_text_val,
#                 X_features_val,
#                 y_val.tolist() if hasattr(y_val, 'tolist') else y_val,
#                 self.tokenizer,
#                 self.max_length
#             )
#             val_loader = DataLoader(
#                 val_dataset,
#                 batch_size=self.batch_size,
#                 shuffle=False
#             )
#
#         # Setup optimizer and scheduler
#         optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
#         total_steps = len(train_loader) * self.epochs
#         scheduler = get_linear_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=int(0.1 * total_steps),
#             num_training_steps=total_steps
#         )
#
#         # Training loop
#         best_val_f1 = 0
#         for epoch in range(self.epochs):
#             print(f"\nEpoch {epoch + 1}/{self.epochs}")
#
#             # Training
#             train_loss, train_acc, train_minority_recall = self._train_epoch(
#                 train_loader, optimizer, scheduler
#             )
#
#             epoch_stats = {
#                 'epoch': epoch + 1,
#                 'train_loss': train_loss,
#                 'train_accuracy': train_acc,
#                 'train_minority_recall': train_minority_recall
#             }
#
#             # Validation
#             if val_loader is not None:
#                 val_loss, val_acc, val_minority_recall, val_f1 = self._evaluate_epoch(val_loader)
#
#                 epoch_stats.update({
#                     'val_loss': val_loss,
#                     'val_accuracy': val_acc,
#                     'val_minority_recall': val_minority_recall,
#                     'val_f1_macro': val_f1
#                 })
#
#                 print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
#                       f"Val Minority Recall: {val_minority_recall:.4f} | Val F1: {val_f1:.4f}")
#
#                 if val_f1 > best_val_f1:
#                     best_val_f1 = val_f1
#                     print(f"✓ New best validation F1: {best_val_f1:.4f}")
#
#             self.training_stats.append(epoch_stats)
#
#         print("\n" + "=" * 80)
#         print("TRAINING COMPLETED")
#         print("=" * 80 + "\n")
#
#         return self
#
#     def _train_epoch(self, train_loader, optimizer, scheduler):
#         """Train for one epoch"""
#         self.model.train()
#         total_loss = 0
#         correct_predictions = 0
#         total_predictions = 0
#         minority_correct = 0
#         minority_total = 0
#
#         progress_bar = tqdm(train_loader, desc="Training")
#
#         for batch in progress_bar:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             manual_features = batch['manual_features'].to(device)
#             labels = batch['label'].to(device)
#
#             # Forward pass
#             logits = self.model(input_ids, attention_mask, manual_features)
#
#             # Calculate loss
#             if self.focal_loss:
#                 loss = self._focal_loss(logits, labels, self.focal_alpha, self.focal_gamma)
#             elif self.use_class_weights:
#                 loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
#                 loss = loss_fct(logits, labels)
#             else:
#                 loss_fct = nn.CrossEntropyLoss()
#                 loss = loss_fct(logits, labels)
#
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#             optimizer.step()
#             scheduler.step()
#
#             # Calculate metrics
#             preds = torch.argmax(logits, dim=1)
#             correct_predictions += torch.sum(preds == labels).item()
#             total_predictions += labels.size(0)
#             total_loss += loss.item()
#
#             # Calculate minority class recall (class 1)
#             minority_mask = (labels == 1)
#             if minority_mask.sum() > 0:
#                 minority_correct += torch.sum((preds == 1) & minority_mask).item()
#                 minority_total += minority_mask.sum().item()
#
#             # Update progress bar
#             progress_bar.set_postfix({
#                 'loss': loss.item(),
#                 'acc': correct_predictions / total_predictions
#             })
#
#         avg_loss = total_loss / len(train_loader)
#         avg_acc = correct_predictions / total_predictions
#         minority_recall = minority_correct / minority_total if minority_total > 0 else 0
#
#         print(f"Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} | "
#               f"Train Minority Recall: {minority_recall:.4f}")
#
#         return avg_loss, avg_acc, minority_recall
#
#     def _evaluate_epoch(self, val_loader):
#         """Evaluate on validation set"""
#         self.model.eval()
#         total_loss = 0
#         all_preds = []
#         all_labels = []
#
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc="Validating"):
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                 manual_features = batch['manual_features'].to(device)
#                 labels = batch['label'].to(device)
#
#                 logits = self.model(input_ids, attention_mask, manual_features)
#
#                 # Calculate loss
#                 if self.focal_loss:
#                     loss = self._focal_loss(logits, labels, self.focal_alpha, self.focal_gamma)
#                 elif self.use_class_weights:
#                     loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
#                     loss = loss_fct(logits, labels)
#                 else:
#                     loss_fct = nn.CrossEntropyLoss()
#                     loss = loss_fct(logits, labels)
#
#                 total_loss += loss.item()
#
#                 preds = torch.argmax(logits, dim=1)
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
#
#         avg_loss = total_loss / len(val_loader)
#
#         # Calculate metrics
#         all_preds = np.array(all_preds)
#         all_labels = np.array(all_labels)
#
#         correct = (all_preds == all_labels).sum()
#         accuracy = correct / len(all_labels)
#
#         # Minority recall
#         minority_mask = (all_labels == 1)
#         minority_recall = 0
#         if minority_mask.sum() > 0:
#             minority_correct = ((all_preds == 1) & minority_mask).sum()
#             minority_recall = minority_correct / minority_mask.sum()
#
#         # F1 score
#         report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
#         f1_macro = report['macro avg']['f1-score']
#
#         return avg_loss, accuracy, minority_recall, f1_macro
#
#     def predict(self, X_text, X_features):
#         """
#         Make predictions on new data
#
#         Args:
#             X_text: Texts to predict (list or Series)
#             X_features: Manual features (numpy array or DataFrame)
#
#         Returns:
#             numpy array of predictions
#         """
#         if self.model is None:
#             raise ValueError("Model not trained. Call fit() first.")
#
#         self.model.eval()
#
#         # Convert features
#         if isinstance(X_features, pd.DataFrame):
#             X_features = X_features.values
#         X_features = np.array(X_features, dtype=np.float32)
#
#         # Create dataset with dummy labels
#         dummy_labels = [0] * len(X_text)
#         dataset = TextFeatureDataset(
#             X_text.tolist() if hasattr(X_text, 'tolist') else X_text,
#             X_features,
#             dummy_labels,
#             self.tokenizer,
#             self.max_length
#         )
#
#         dataloader = DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=False
#         )
#
#         predictions = []
#
#         with torch.no_grad():
#             for batch in tqdm(dataloader, desc="Predicting"):
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                 manual_features = batch['manual_features'].to(device)
#
#                 logits = self.model(input_ids, attention_mask, manual_features)
#                 preds = torch.argmax(logits, dim=1)
#                 predictions.extend(preds.cpu().numpy())
#
#         return np.array(predictions)
#
#     def predict_proba(self, X_text, X_features):
#         """
#         Get prediction probabilities
#
#         Args:
#             X_text: Texts to predict (list or Series)
#             X_features: Manual features (numpy array or DataFrame)
#
#         Returns:
#             numpy array of shape (n_samples, 2) with probabilities
#         """
#         if self.model is None:
#             raise ValueError("Model not trained. Call fit() first.")
#
#         self.model.eval()
#
#         # Convert features
#         if isinstance(X_features, pd.DataFrame):
#             X_features = X_features.values
#         X_features = np.array(X_features, dtype=np.float32)
#
#         # Create dataset
#         dummy_labels = [0] * len(X_text)
#         dataset = TextFeatureDataset(
#             X_text.tolist() if hasattr(X_text, 'tolist') else X_text,
#             X_features,
#             dummy_labels,
#             self.tokenizer,
#             self.max_length
#         )
#
#         dataloader = DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=False
#         )
#
#         probabilities = []
#
#         with torch.no_grad():
#             for batch in tqdm(dataloader, desc="Getting probabilities"):
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                 manual_features = batch['manual_features'].to(device)
#
#                 logits = self.model(input_ids, attention_mask, manual_features)
#                 probs = torch.softmax(logits, dim=1)
#                 probabilities.extend(probs.cpu().numpy())
#
#         return np.array(probabilities)
#
#     def find_optimal_threshold(self, X_text_val, X_features_val, y_val):
#         """Find optimal classification threshold on validation set"""
#         print("\nFinding optimal classification threshold...")
#
#         y_proba = self.predict_proba(X_text_val, X_features_val)
#         y_proba_class1 = y_proba[:, 1]
#
#         best_f1 = 0
#         best_threshold = 0.5
#
#         for threshold in np.arange(0.3, 0.8, 0.05):
#             y_pred = (y_proba_class1 >= threshold).astype(int)
#             report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
#             f1 = report['macro avg']['f1-score']
#
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_threshold = threshold
#
#         self.prediction_threshold = best_threshold
#         print(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
#
#         return best_threshold
#
#     def evaluate(self, X_text_test, X_features_test, y_test):
#         """
#         Evaluate model on test set
#
#         Args:
#             X_text_test: Test texts
#             X_features_test: Test manual features
#             y_test: Test labels
#
#         Returns:
#             Tuple of (results_dict, predictions, probabilities)
#         """
#         print("\n" + "=" * 80)
#         print("EVALUATING BERT FEATURE FUSION MODEL")
#         print("=" * 80)
#
#         y_pred = self.predict(X_text_test, X_features_test)
#         y_proba = self.predict_proba(X_text_test, X_features_test)
#
#         # Get classification report
#         report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
#
#         results = {
#             'model': 'bert_feature_fusion',
#             'test_f1_macro': report['macro avg']['f1-score'],
#             'test_accuracy': report['accuracy'],
#             'test_precision': report['macro avg']['precision'],
#             'test_recall': report['macro avg']['recall'],
#             'minority_recall': report['1']['recall'],
#             'minority_precision': report['1']['precision'],
#             'minority_f1': report['1']['f1-score']
#         }
#
#         print("\nTest Results:")
#         print(f"F1 Score (Macro): {results['test_f1_macro']:.4f}")
#         print(f"Accuracy: {results['test_accuracy']:.4f}")
#         print(f"Precision: {results['test_precision']:.4f}")
#         print(f"Recall: {results['test_recall']:.4f}")
#         print(f"\nMinority Class (Class 1) Performance:")
#         print(f"Recall: {results['minority_recall']:.4f}")
#         print(f"Precision: {results['minority_precision']:.4f}")
#         print(f"F1: {results['minority_f1']:.4f}")
#
#         print("\nDetailed Classification Report:")
#         print(classification_report(y_test, y_pred, zero_division=0))
#
#         print("=" * 80 + "\n")
#
#         return results, y_pred, y_proba
#
#     def get_training_stats(self):
#         """Return training statistics as DataFrame"""
#         return pd.DataFrame(self.training_stats)
#
#     def save_model(self, path):
#         """Save model to disk"""
#         if self.model is None:
#             raise ValueError("No model to save. Train the model first.")
#
#         torch.save({
#             'model_state_dict': self.model.state_dict(),
#             'num_manual_features': self.num_manual_features,
#             'prediction_threshold': self.prediction_threshold,
#             'class_weights': self.class_weights
#         }, f"{path}/model.pt")
#
#         self.tokenizer.save_pretrained(path)
#         print(f"Model saved to {path}")
#
#     def load_model(self, path):
#         """Load model from disk"""
#         checkpoint = torch.load(f"{path}/model.pt")
#
#         self.num_manual_features = checkpoint['num_manual_features']
#         self.prediction_threshold = checkpoint['prediction_threshold']
#         self.class_weights = checkpoint['class_weights']
#
#         self.model = self._create_model(self.num_manual_features)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.model.eval()
#
#         self.tokenizer = BertTokenizer.from_pretrained(path)
#         print(f"Model loaded from {path}")
#
#
# # ============================================================================
# # WRAPPER FUNCTION FOR EASY TRAINING
# # ============================================================================
#
# def train_fusion_model(X_text_train, X_features_train, y_train,
#                        X_text_test, X_features_test, y_test,
#                        model_name='bert-base-uncased',
#                        max_length=256,
#                        batch_size=4,
#                        learning_rate=2e-5,
#                        epochs=5,
#                        use_class_weights=True,
#                        use_balanced_sampler=True,
#                        focal_loss=False):
#     """
#     Wrapper function to train BERT Feature Fusion model
#
#     Args:
#         X_text_train: Training text data
#         X_features_train: Training manual features
#         y_train: Training labels
#         X_text_test: Test text data
#         X_features_test: Test manual features
#         y_test: Test labels
#         model_name: BERT model name
#         max_length: Maximum sequence length
#         batch_size: Batch size for training
#         learning_rate: Learning rate
#         epochs: Number of epochs
#         use_class_weights: Whether to use class weights
#         use_balanced_sampler: Whether to use balanced sampling
#         focal_loss: Whether to use focal loss
#
#     Returns:
#         Tuple of (trained_model, results_dict)
#     """
#     from sklearn.model_selection import train_test_split
#
#     print("\n" + "=" * 80)
#     print("BERT FEATURE FUSION MODEL - TRAINING WRAPPER")
#     print("=" * 80)
#
#     # Create the classifier
#     fusion_classifier = BERTFeatureFusionClassifier(
#         model_name=model_name,
#         max_length=max_length,
#         batch_size=batch_size,
#         learning_rate=learning_rate,
#         epochs=epochs,
#         random_state=42,
#         use_class_weights=use_class_weights,
#         use_balanced_sampler=use_balanced_sampler,
#         focal_loss=focal_loss
#     )
#
#     # Split training data into train/val
#     X_text_tr, X_text_val, X_feat_tr, X_feat_val, y_tr, y_val = train_test_split(
#         X_text_train, X_features_train, y_train,
#         test_size=0.2,
#         random_state=42,
#         stratify=y_train
#     )
#
#     print(f"Train size: {len(X_text_tr)}")
#     print(f"Val size: {len(X_text_val)}")
#     print(f"Test size: {len(X_text_test)}")
#     print(f"Manual features: {X_features_train.shape[1]}")
#
#     # Train the model
#     fusion_classifier.fit(
#         X_text_tr, X_feat_tr, y_tr,
#         X_text_val, X_feat_val, y_val
#     )
#
#     # Find optimal threshold
#     fusion_classifier.find_optimal_threshold(X_text_val, X_feat_val, y_val)
#
#     # Evaluate on test set
#     results, y_pred, y_proba = fusion_classifier.evaluate(
#         X_text_test, X_features_test, y_test
#     )
#
#     # Display training statistics
#     print("\n" + "=" * 80)
#     print("TRAINING STATISTICS")
#     print("=" * 80)
#     training_stats = fusion_classifier.get_training_stats()
#     print(training_stats.to_string())
#     print("=" * 80 + "\n")
#
#     # Generate visualizations
#     try:
#         from vis import generate_fusion_plots  # Changed from generate_bert_plots
#
#         # FIXED: Call with BOTH text and features
#         y_pred_vis, y_proba_vis = generate_fusion_plots(
#             fusion_classifier,
#             X_text_test,
#             X_features_test,  # <-- THIS WAS MISSING! This is the key fix
#             y_test,
#             model_name='Feature Fusion',
#             save_dir='visualizations'
#         )
#
#     except ImportError:
#         print("Warning: vis.py not found. Skipping visualizations.")
#     except Exception as e:
#         print(f"Warning: Could not generate visualizations: {e}")
#         import traceback
#         traceback.print_exc()
#
#     print("\n" + "=" * 80)
#     print("TRAINING COMPLETE!")
#     print("=" * 80 + "\n")
#
#     return fusion_classifier, results
#
#
# # Example usage (if run directly):
# if __name__ == "__main__":
#     print("Feature Fusion Module Loaded Successfully!")
#     print("\nTo train a fusion model:")
#     print("  from feature_fusion import train_fusion_model")
#     print("  model, results = train_fusion_model(X_text_train, X_feat_train, y_train,")
#     print("                                      X_text_test, X_feat_test, y_test)")
#
#
