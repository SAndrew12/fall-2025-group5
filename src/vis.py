import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve

def plot_model_performance(results_df, metric='f1_macro'):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='model', y=metric, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Model Comparison on {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_best(trainer, X_test, y_test):
    model, stats = trainer.get_best_model()
    X_test_proc = trainer._apply_preprocessing(X_test, fit=False)
    y_pred = model.predict(X_test_proc)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    disp.ax_.set_title(f'Confusion Matrix: {stats["model"]}')
    plt.tight_layout()
    plt.show()

def plot_roc_pr(trainer, X_test, y_test):
    model, stats = trainer.get_best_model()
    model_name = stats["model"]
    X_test_proc = trainer._apply_preprocessing(X_test, fit=False)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_proc)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_proc)
    else:
        print(f"{model_name} does not support probability estimates.")
        return

    fpr, tpr, _ = roc_curve(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.figure(figsize=(12, 5))

    # ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f"ROC Curve: {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    # PR Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.tight_layout()
    plt.show()
