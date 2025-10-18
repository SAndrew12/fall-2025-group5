import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set device
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


class BERTClassifier:
    """BERT-based binary classifier"""

    def __init__(self, model_name='bert-base-uncased', max_length=128,
                 batch_size=16, learning_rate=2e-5, epochs=3, random_state=42):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        self.training_stats = []

    def _create_model(self):
        """Create a fresh BERT model"""


        from torch import tensor
        class_weights = tensor([1.0, 49.0]).to(device)

        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )

        self.class_weights = class_weights
        return model.to(device)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the BERT model

        Args:
            X_train: Training texts (list or Series)
            y_train: Training labels (list or Series)
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        print("\n" + "=" * 60)
        print("TRAINING BERT CLASSIFIER")
        print("=" * 60)

        # Create model
        self.model = self._create_model()

        # Create datasets
        train_dataset = TextDataset(
            X_train.tolist() if hasattr(X_train, 'tolist') else X_train,
            y_train.tolist() if hasattr(y_train, 'tolist') else y_train,
            self.tokenizer,
            self.max_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Create validation loader if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TextDataset(
                X_val.tolist() if hasattr(X_val, 'tolist') else X_val,
                y_val.tolist() if hasattr(y_val, 'tolist') else y_val,
                self.tokenizer,
                self.max_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Training loop
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Training
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, scheduler
            )

            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc
            }

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._evaluate_epoch(val_loader)
                epoch_stats['val_loss'] = val_loss
                epoch_stats['val_accuracy'] = val_acc
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

            self.training_stats.append(epoch_stats)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        return self

    def _train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)
            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct_predictions / total_predictions
            })

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct_predictions / total_predictions

        return avg_loss, avg_acc

    def _evaluate_epoch(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1)
                correct_predictions += torch.sum(preds == labels).item()
                total_predictions += labels.size(0)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        avg_acc = correct_predictions / total_predictions

        return avg_loss, avg_acc

    def predict(self, X):
        """
        Make predictions on new data

        Args:
            X: Texts to predict (list or Series)

        Returns:
            numpy array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()

        # Create dataset
        # Use dummy labels (0s) since we only need predictions
        dummy_labels = [0] * len(X)
        dataset = TextDataset(
            X.tolist() if hasattr(X, 'tolist') else X,
            dummy_labels,
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

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Get prediction probabilities

        Args:
            X: Texts to predict (list or Series)

        Returns:
            numpy array of shape (n_samples, 2) with probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()

        # Create dataset
        dummy_labels = [0] * len(X)
        dataset = TextDataset(
            X.tolist() if hasattr(X, 'tolist') else X,
            dummy_labels,
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

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set

        Args:
            X_test: Test texts
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATING BERT MODEL")
        print("=" * 60)

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results = {
            'model': 'bert',
            'test_f1_macro': report['macro avg']['f1-score'],
            'test_accuracy': report['accuracy'],
            'test_precision': report['macro avg']['precision'],
            'test_recall': report['macro avg']['recall']
        }

        print("\nTest Results:")
        print(f"F1 Score (Macro): {results['test_f1_macro']:.4f}")
        print(f"Accuracy: {results['test_accuracy']:.4f}")
        print(f"Precision: {results['test_precision']:.4f}")
        print(f"Recall: {results['test_recall']:.4f}")

        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("=" * 60 + "\n")

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