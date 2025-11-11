import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TextDataset(Dataset):
    """Dataset for text classification with dynamic padding support"""

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
    """
    BERT-based binary classifier with improved regularization and training strategies

    Key improvements:
    - Configurable layer freezing and regularization
    - Choice between class weights OR balanced sampler (not both)
    - Gradient accumulation for effective larger batch sizes
    - Learning rate finder support
    - Dynamic text length analysis
    """

    def __init__(self, model_name='bert-base-uncased', max_length=128,
                 batch_size=16, learning_rate=2e-5, epochs=3, random_state=42,
                 class_imbalance_strategy='balanced_sampler',  # NEW: 'class_weights', 'balanced_sampler', or 'none'
                 focal_loss=False, focal_alpha=0.25, focal_gamma=2.0,
                 prediction_threshold=0.5,
                 freeze_bert_base=True,
                 unfreeze_last_n_layers=2,  # REDUCED from 4 to 2
                 label_smoothing=0.1,
                 dropout_rate=0.5,  # NEW: Increased dropout
                 gradient_accumulation_steps=1,  # NEW: For effective larger batch size
                 beta_class_weights=0.99):  # NEW: More moderate class weighting

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.class_imbalance_strategy = class_imbalance_strategy
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.prediction_threshold = prediction_threshold
        self.freeze_bert_base = freeze_bert_base
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.label_smoothing = label_smoothing
        self.dropout_rate = dropout_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.beta_class_weights = beta_class_weights

        # Validation
        if class_imbalance_strategy not in ['class_weights', 'balanced_sampler', 'none']:
            raise ValueError("class_imbalance_strategy must be 'class_weights', 'balanced_sampler', or 'none'")

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Add special tokens for semantic placeholders
        special_tokens = {'additional_special_tokens': ['[ARMED_GROUP]', '[LOCATION]']}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens to tokenizer")

        self.model = None
        self.training_stats = []
        self.class_weights = None

    def analyze_text_lengths(self, texts):
        """
        Analyze text lengths to suggest optimal max_length

        Args:
            texts: List or Series of text strings

        Returns:
            Dictionary with statistics
        """
        print("\n" + "=" * 60)
        print("ANALYZING TEXT LENGTHS")
        print("=" * 60)

        lengths = []
        for text in tqdm(texts, desc="Tokenizing texts"):
            tokens = self.tokenizer.tokenize(str(text))
            lengths.append(len(tokens))

        lengths = np.array(lengths)
        stats = {
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'std': np.std(lengths),
            '95th_percentile': np.percentile(lengths, 95),
            '99th_percentile': np.percentile(lengths, 99),
            'max': np.max(lengths)
        }

        print(f"\nText Length Statistics (in tokens):")
        print(f"  Mean: {stats['mean']:.1f}")
        print(f"  Median: {stats['median']:.1f}")
        print(f"  Std Dev: {stats['std']:.1f}")
        print(f"  95th percentile: {stats['95th_percentile']:.1f}")
        print(f"  99th percentile: {stats['99th_percentile']:.1f}")
        print(f"  Max: {stats['max']}")
        print(f"\nCurrent max_length: {self.max_length}")

        suggested_max_length = int(stats['95th_percentile']) + 10  # Add buffer
        print(f"Suggested max_length: {suggested_max_length}")

        if self.max_length > suggested_max_length * 1.5:
            print(f"⚠️  WARNING: Current max_length ({self.max_length}) is much larger than needed!")
            print(f"   Consider using max_length={suggested_max_length} for efficiency")

        print("=" * 60 + "\n")
        return stats

    def _calculate_class_weights(self, y_train):
        """Calculate balanced class weights using effective number of samples"""
        unique, counts = np.unique(y_train, return_counts=True)

        # Effective number of samples (moderate weighting with configurable beta)
        beta = self.beta_class_weights
        effective_num = 1.0 - np.power(beta, counts)
        weights_effective = (1.0 - beta) / effective_num
        weights_effective = weights_effective / weights_effective.sum() * len(unique)

        class_weights = torch.FloatTensor(weights_effective).to(device)

        print(f"\nClass distribution: {dict(zip(unique, counts))}")
        print(f"Class weights (beta={beta}): {dict(zip(unique, weights_effective))}")

        return class_weights

    def _create_model(self):
        """Create a BERT model with improved regularization"""
        # Load model with higher dropout
        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            hidden_dropout_prob=self.dropout_rate,
            attention_probs_dropout_prob=self.dropout_rate,
            output_attentions=False,
            output_hidden_states=False
        )

        # Resize token embeddings to account for new special tokens
        model.resize_token_embeddings(len(self.tokenizer))

        # Optionally freeze most of the BERT base
        if self.freeze_bert_base:
            # Freeze all BERT encoder params
            for param in model.bert.parameters():
                param.requires_grad = False

            # Unfreeze last N encoder layers
            if self.unfreeze_last_n_layers is not None and self.unfreeze_last_n_layers > 0:
                for layer in model.bert.encoder.layer[-self.unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

            # Always keep the classification head trainable
            for param in model.classifier.parameters():
                param.requires_grad = True

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"\nModel Architecture:"
                f"\n  Total parameters: {total_params:,}"
                f"\n  Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})"
                f"\n  Frozen base with {self.unfreeze_last_n_layers} unfrozen layers"
            )

        return model.to(device)

    def _create_balanced_sampler(self, y_train):
        """Create a weighted sampler for balanced batches"""
        class_counts = np.bincount(y_train)
        class_weights = 1. / class_counts
        sample_weights = class_weights[y_train]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler

    def _focal_loss(self, logits, labels, alpha=0.25, gamma=2.0):
        """
        Focal Loss for handling class imbalance
        Focuses training on hard examples
        """
        ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the BERT model with improved training strategy

        Args:
            X_train: Training texts (list or Series)
            y_train: Training labels (list or Series)
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        print("\n" + "=" * 60)
        print("TRAINING BERT CLASSIFIER")
        print("=" * 60)
        print(f"Class Imbalance Strategy: {self.class_imbalance_strategy}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {self.batch_size * self.gradient_accumulation_steps}")
        print("=" * 60)

        # Convert to lists if needed
        X_train_list = X_train.tolist() if hasattr(X_train, 'tolist') else X_train
        y_train_list = y_train.tolist() if hasattr(y_train, 'tolist') else y_train
        y_train_array = np.array(y_train_list)

        # Calculate class weights if using that strategy
        if self.class_imbalance_strategy == 'class_weights':
            self.class_weights = self._calculate_class_weights(y_train_array)

        # Create model
        self.model = self._create_model()

        # Create datasets
        train_dataset = TextDataset(
            X_train_list,
            y_train_list,
            self.tokenizer,
            self.max_length
        )

        # Create sampler if using balanced sampling
        sampler = None
        shuffle = True
        if self.class_imbalance_strategy == 'balanced_sampler':
            sampler = self._create_balanced_sampler(y_train_array)
            shuffle = False  # Sampler handles shuffling
            print("Using balanced batch sampler")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler
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
        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
            num_training_steps=total_steps
        )

        # Training loop
        best_val_f1 = 0
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Training
            train_loss, train_acc, train_minority_recall = self._train_epoch(
                train_loader, optimizer, scheduler
            )

            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_minority_recall': train_minority_recall
            }

            # Validation
            if val_loader is not None:
                val_loss, val_acc, val_minority_recall, val_f1 = self._evaluate_epoch(val_loader)
                epoch_stats['val_loss'] = val_loss
                epoch_stats['val_accuracy'] = val_acc
                epoch_stats['val_minority_recall'] = val_minority_recall
                epoch_stats['val_f1'] = val_f1

                print(
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Train Min Recall: {train_minority_recall:.4f}"
                )
                print(
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"Val Min Recall: {val_minority_recall:.4f} | Val F1 (macro): {val_f1:.4f}"
                )

                # Track best model based on validation F1
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {
                        "model": self.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                    print(f"New best validation F1: {val_f1:.4f}")
            else:
                print(
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Train Min Recall: {train_minority_recall:.4f}"
                )

            self.training_stats.append(epoch_stats)

        # Load best validation checkpoint if we recorded one
        if 'best_state' in locals():
            self.model.load_state_dict(best_state["model"])
            print("\nLoaded best validation checkpoint.")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        return self

    def _train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        minority_correct = 0
        minority_total = 0

        progress_bar = tqdm(train_loader, desc="Training")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

            # Calculate loss based on configuration
            if self.focal_loss:
                loss = self._focal_loss(logits, labels, self.focal_alpha, self.focal_gamma)
            else:
                if self.class_imbalance_strategy == 'class_weights' and self.class_weights is not None:
                    loss_fct = torch.nn.CrossEntropyLoss(
                        weight=self.class_weights,
                        label_smoothing=self.label_smoothing
                    )
                else:
                    loss_fct = torch.nn.CrossEntropyLoss(
                        label_smoothing=self.label_smoothing
                    )
                loss = loss_fct(logits, labels)

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Calculate metrics
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

            # Track minority class (assuming class 1 is minority)
            minority_mask = labels == 1
            if minority_mask.sum() > 0:
                minority_correct += torch.sum((preds == labels) & minority_mask).item()
                minority_total += minority_mask.sum().item()

            total_loss += loss.item() * self.gradient_accumulation_steps

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() * self.gradient_accumulation_steps,
                'acc': correct_predictions / total_predictions
            })

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct_predictions / total_predictions
        minority_recall = minority_correct / minority_total if minority_total > 0 else 0

        return avg_loss, avg_acc, minority_recall

    def _evaluate_epoch(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        minority_correct = 0
        minority_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits

                # Calculate loss
                if self.focal_loss:
                    loss = self._focal_loss(logits, labels, self.focal_alpha, self.focal_gamma)
                else:
                    if self.class_imbalance_strategy == 'class_weights' and self.class_weights is not None:
                        loss_fct = torch.nn.CrossEntropyLoss(
                            weight=self.class_weights,
                            label_smoothing=self.label_smoothing
                        )
                    else:
                        loss_fct = torch.nn.CrossEntropyLoss(
                            label_smoothing=self.label_smoothing
                        )
                    loss = loss_fct(logits, labels)

                preds = torch.argmax(logits, dim=1)
                correct_predictions += torch.sum(preds == labels).item()
                total_predictions += labels.size(0)

                # Track minority class
                minority_mask = labels == 1
                if minority_mask.sum() > 0:
                    minority_correct += torch.sum((preds == labels) & minority_mask).item()
                    minority_total += minority_mask.sum().item()

                total_loss += loss.item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        avg_acc = correct_predictions / total_predictions
        minority_recall = minority_correct / minority_total if minority_total > 0 else 0

        # Calculate F1 score (macro)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, avg_acc, minority_recall, f1

    def predict(self, X):
        """
        Make predictions on new data with adjustable threshold

        Args:
            X: Texts to predict (list or Series)

        Returns:
            numpy array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        probabilities = self.predict_proba(X)

        # Apply custom threshold for minority class (class 1)
        predictions = (probabilities[:, 1] >= self.prediction_threshold).astype(int)

        return predictions

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

    def find_optimal_threshold(self, X_val, y_val):
        """
        Find optimal prediction threshold for minority class

        Args:
            X_val: Validation texts
            y_val: Validation labels

        Returns:
            optimal_threshold: Best threshold for F1 score (class 1)
        """
        probabilities = self.predict_proba(X_val)

        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5

        print("\nFinding optimal threshold (minority F1)...")
        for threshold in thresholds:
            preds = (probabilities[:, 1] >= threshold).astype(int)
            # F1 specifically for class 1
            f1_minority = f1_score(y_val, preds, pos_label=1)
            if f1_minority > best_f1:
                best_f1 = f1_minority
                best_threshold = threshold

        print(f"Optimal threshold: {best_threshold:.2f} (Minority F1: {best_f1:.4f})")
        self.prediction_threshold = best_threshold

        return best_threshold

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
            'test_recall': report['macro avg']['recall'],
            'minority_precision': report['1']['precision'],
            'minority_recall': report['1']['recall'],
            'minority_f1': report['1']['f1-score']
        }

        print("\nTest Results:")
        print(f"F1 Score (Macro): {results['test_f1_macro']:.4f}")
        print(f"Accuracy: {results['test_accuracy']:.4f}")
        print(f"Precision: {results['test_precision']:.4f}")
        print(f"Recall: {results['test_recall']:.4f}")
        print(f"\nMinority Class (Class 1):")
        print(f"  Precision: {results['minority_precision']:.4f}")
        print(f"  Recall: {results['minority_recall']:.4f}")
        print(f"  F1-Score: {results['minority_f1']:.4f}")

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


# ============================================================================
# LEARNING RATE FINDER
# ============================================================================

def find_learning_rate(model, train_loader, optimizer_class=AdamW,
                       start_lr=1e-7, end_lr=1, num_iter=100,
                       beta=0.98, device='cuda'):
    """
    Learning rate finder using the method from
    "Cyclical Learning Rates for Training Neural Networks" (Smith, 2017)

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        optimizer_class: Optimizer class (default: AdamW)
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iter: Number of iterations
        beta: Smoothing factor for losses
        device: Device to run on

    Returns:
        lrs: List of learning rates tested
        losses: List of corresponding losses
        suggested_lr: Suggested learning rate
    """
    print("\n" + "=" * 60)
    print("LEARNING RATE FINDER")
    print("=" * 60)

    model.train()

    # Save initial state
    initial_state = {
        'model': model.state_dict(),
    }

    # Initialize optimizer with start_lr
    optimizer = optimizer_class(model.parameters(), lr=start_lr)

    # Learning rate schedule
    gamma = (end_lr / start_lr) ** (1 / num_iter)

    lrs = []
    losses = []
    best_loss = float('inf')
    avg_loss = 0.0

    iterator = iter(train_loader)

    for iteration in tqdm(range(num_iter), desc="Finding LR"):
        # Get batch
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)

        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (iteration + 1))

        # Record
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(smoothed_loss)

        # Check if loss is exploding
        if iteration > 0 and smoothed_loss > 4 * best_loss:
            print(f"\nStopping early at iteration {iteration} - loss is exploding")
            break

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma

    # Restore model to initial state
    model.load_state_dict(initial_state['model'])

    # Find suggested LR (steepest descent point)
    try:
        min_grad_idx = np.gradient(np.array(losses)).argmin()
        suggested_lr = lrs[min_grad_idx]
    except:
        suggested_lr = lrs[len(lrs) // 3]  # Fallback to 1/3 point

    print(f"\nSuggested learning rate: {suggested_lr:.2e}")
    print(f"(LR at steepest gradient descent)")
    print("=" * 60 + "\n")

    return lrs, losses, suggested_lr


def plot_lr_finder(lrs, losses, suggested_lr=None, save_path='lr_finder_plot.png'):
    """
    Plot learning rate finder results

    Args:
        lrs: List of learning rates
        losses: List of losses
        suggested_lr: Suggested learning rate to mark on plot
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True, alpha=0.3)

    if suggested_lr:
        plt.axvline(x=suggested_lr, color='r', linestyle='--',
                    label=f'Suggested LR: {suggested_lr:.2e}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()