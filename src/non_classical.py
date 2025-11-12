"""
BERT Text Classification Model
Simple and clean implementation for binary classification
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
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


class TextDataset(Dataset):
    """Dataset for text classification"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

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
            'label': torch.tensor(label, dtype=torch.long)
        }


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()


class BERTClassifier:
    """BERT classifier for binary text classification"""

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
                 prediction_threshold=0.5):
        """
        Initialize BERT classifier

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

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        self.training_stats = []

    def _create_model(self):
        """Create BERT model"""
        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            hidden_dropout_prob=self.dropout_rate,
            attention_probs_dropout_prob=self.dropout_rate,
            output_attentions=False,
            output_hidden_states=False
        )

        if self.freeze_bert_base:
            # Freeze all BERT parameters
            for param in model.bert.parameters():
                param.requires_grad = False

            # Unfreeze last N layers
            if self.unfreeze_last_n_layers > 0:
                for layer in model.bert.encoder.layer[-self.unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

            # Keep classifier trainable
            for param in model.classifier.parameters():
                param.requires_grad = True

        return model.to(device)

    def fit(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        print("\n" + "=" * 80)
        print("TRAINING BERT MODEL")
        print("=" * 80)

        # Create datasets
        train_dataset = TextDataset(
            X_train.tolist(),
            y_train.tolist(),
            self.tokenizer,
            self.max_length
        )

        val_dataset = TextDataset(
            X_val.tolist(),
            y_val.tolist(),
            self.tokenizer,
            self.max_length
        )

        # Create data loaders
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
        self.model = self._create_model()

        # Loss function
        if self.use_focal_loss:
            criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss()

        # Optimizer and scheduler
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01
        )

        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Early stopping setup
        best_val_metric = 0
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Training phase
            self.model.train()
            train_loss = 0
            optimizer.zero_grad()

            for i, batch in enumerate(tqdm(train_loader, desc="Training")):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss = loss / self.gradient_accumulation_steps

                loss.backward()
                train_loss += loss.item()

                if (i + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

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
                    labels = batch['label'].to(device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs.logits, dim=1)
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

            # Early stopping check
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

    def predict(self, X_test):
        """Make predictions on test data"""
        self.model.eval()

        dataset = TextDataset(
            X_test.tolist(),
            [0] * len(X_test),
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

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=1)
                preds = (probs[:, 1] >= self.prediction_threshold).long()
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        self.model.eval()

        dataset = TextDataset(
            X_test.tolist(),
            [0] * len(X_test),
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

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def find_optimal_threshold(self, X_val, y_val):
        """Find optimal classification threshold on validation set"""
        print("\nFinding optimal classification threshold...")

        y_proba = self.predict_proba(X_val)
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

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        print("\n" + "=" * 80)
        print("EVALUATING BERT MODEL")
        print("=" * 80)

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results = {
            'model': 'bert',
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

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model from disk"""
        self.model = BertForSequenceClassification.from_pretrained(path).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    print("BERT Classifier Module Loaded Successfully!")
    print("\nTo train a BERT model:")
    print("  from non_classical import BERTClassifier")
    print("  model = BERTClassifier()")
    print("  model.fit(X_train, y_train, X_val, y_val)")