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
                 max_length=320,
                 batch_size=16,
                 learning_rate=1e-5,
                 epochs=12,
                 random_state=42,
                 early_stopping_patience=3,
                 focal_loss=True,
                 focal_alpha=0.7,
                 focal_gamma=2.5,
                 freeze_bert_base=False,
                 unfreeze_last_n_layers=12,
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
            for param in model.fusion.parameters():
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
            val_f1 = report['1']['f1-score']
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

        for threshold in np.arange(0.1, 0.9, 0.02):
            y_pred = (y_proba_class1 >= threshold).astype(int)
            report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            f1 = report['1']['f1-score']  # optimize minority F1

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.prediction_threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

        return best_threshold

    def evaluate(self, X_text_test, X_features_test, y_test):
        """
        Evaluate model on test set with standardized metrics

        This version outputs the same metrics as Classical and BERT models
        for consistent comparison across all model types.
        """
        from sklearn.metrics import confusion_matrix, roc_auc_score

        print("\n" + "=" * 80)
        print("EVALUATING BERT FEATURE FUSION MODEL")
        print("=" * 80)

        y_pred = self.predict(X_text_test, X_features_test)
        y_proba = self.predict_proba(X_text_test, X_features_test)

        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # === STANDARDIZED METRICS OUTPUT ===

        results = {
            'model': 'bert_feature_fusion',

            # Macro metrics
            'test_f1_macro': report['macro avg']['f1-score'],
            'test_accuracy': report['accuracy'],
            'test_precision': report['macro avg']['precision'],
            'test_recall': report['macro avg']['recall'],

            # Minority class (Class 1) metrics
            'minority_recall': report['1']['recall'] if '1' in report else 0.0,
            'minority_precision': report['1']['precision'] if '1' in report else 0.0,
            'minority_f1': report['1']['f1-score'] if '1' in report else 0.0,

            # Majority class (Class 0) metrics
            'majority_recall': report['0']['recall'] if '0' in report else 0.0,
            'majority_precision': report['0']['precision'] if '0' in report else 0.0,
            'majority_f1': report['0']['f1-score'] if '0' in report else 0.0,
        }

        # ROC AUC score
        try:
            if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                results['roc_auc_score'] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                results['roc_auc_score'] = roc_auc_score(y_test, y_proba)
        except:
            results['roc_auc_score'] = np.nan

        # Confusion matrix breakdown
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results['true_positives'] = int(tp)
        results['true_negatives'] = int(tn)
        results['false_positives'] = int(fp)
        results['false_negatives'] = int(fn)

        # CV score (N/A for Feature Fusion, but include for consistency)
        results['cv_score'] = np.nan

        # Print results
        print("\nTest Results:")
        print(f"F1 Score (Macro): {results['test_f1_macro']:.4f}")
        print(f"Accuracy: {results['test_accuracy']:.4f}")
        print(f"Precision: {results['test_precision']:.4f}")
        print(f"Recall: {results['test_recall']:.4f}")
        print(f"ROC AUC: {results['roc_auc_score']:.4f}")

        print(f"\nMinority Class (Class 1) Performance:")
        print(f"Recall: {results['minority_recall']:.4f}")
        print(f"Precision: {results['minority_precision']:.4f}")
        print(f"F1: {results['minority_f1']:.4f}")

        print(f"\nMajority Class (Class 0) Performance:")
        print(f"Recall: {results['majority_recall']:.4f}")
        print(f"Precision: {results['majority_precision']:.4f}")
        print(f"F1: {results['majority_f1']:.4f}")

        print(f"\nConfusion Matrix Breakdown:")
        print(f"True Positives: {results['true_positives']}")
        print(f"True Negatives: {results['true_negatives']}")
        print(f"False Positives: {results['false_positives']}")
        print(f"False Negatives: {results['false_negatives']}")

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



