"""
Script đánh giá model trên test set
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

from config import BREED_NAMES, PLOTS_DIR, METRICS_DIR, get_test_dir
from data_loader import create_dataset_from_directory
from utils import load_class_names


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Vẽ confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix đã lưu tại: {save_path}")
    
    plt.show()


def evaluate_model(model_path, batch_size=16):
    """Đánh giá model trên test set"""
    
    print("\n" + "="*60)
    print("ĐÁNH GIÁ MODEL")
    print("="*60 + "\n")
    
    # Load model
    print(f"Loading model từ: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded!\n")
    
    # Load test dataset
    print("Loading test dataset...")
    test_ds = create_dataset_from_directory(
        get_test_dir(),
        is_training=False,
        batch_size=batch_size
    )
    print("Dataset loaded!\n")
    
    # Evaluate
    print("Đang đánh giá...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}\n")
    
    # Predictions
    print("Đang dự đoán...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60 + "\n")
    
    report = classification_report(
        y_true, y_pred,
        target_names=BREED_NAMES,
        digits=4
    )
    print(report)
    
    # Save report
    report_path = METRICS_DIR / 'classification_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report đã lưu tại: {report_path}\n")
    
    # Confusion matrix
    print("Tạo confusion matrix...")
    cm_path = PLOTS_DIR / 'confusion_matrix.png'
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, BREED_NAMES, cm_path)
    
    # Per-class accuracy
    print("\n" + "="*60)
    print("PER-CLASS ACCURACY")
    print("="*60 + "\n")
    
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    for breed, acc in zip(BREED_NAMES, per_class_acc):
        print(f"{breed:25s}: {acc*100:5.2f}%")
    
    # Save metrics
    metrics = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'per_class_accuracy': {breed: float(acc) for breed, acc in zip(BREED_NAMES, per_class_acc)},
        'confusion_matrix': cm.tolist()
    }
    
    metrics_path = METRICS_DIR / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics đã lưu tại: {metrics_path}")
    
    print("\n" + "="*60)
    print("ĐÁNH GIÁ HOÀN TẤT!")
    print("="*60 + "\n")
    
    return metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Đánh giá model')
    parser.add_argument('--model', type=str, required=True, help='Đường dẫn đến model (.keras)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.batch_size)
