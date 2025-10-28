"""
Script training model
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import tensorflow as tf
from tensorflow import keras
from datetime import datetime

from config import (
    EPOCHS, LEARNING_RATE, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR, MIN_LR, FINAL_MODEL_DIR, LOGS_DIR
)
from data_loader import load_datasets
from model import DogBreedClassifier
from utils import save_class_names, save_model_info, save_training_history, export_to_tflite


def create_callbacks():
    """Tạo callbacks cho training"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ModelCheckpoint
    checkpoint_path = str(FINAL_MODEL_DIR / f'best_model_{timestamp}.keras')
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # EarlyStopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # ReduceLROnPlateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=1
    )
    
    # TensorBoard
    log_dir = str(LOGS_DIR / f'tensorboard_{timestamp}')
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    # CSV Logger
    csv_path = str(LOGS_DIR / f'training_log_{timestamp}.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_path)
    
    return [checkpoint_callback, early_stopping, reduce_lr, tensorboard, csv_logger]


def train_model():
    """Main training function"""
    
    print("\n" + "="*60)
    print("DOG BREED CLASSIFICATION - TRAINING")
    print("="*60 + "\n")
    
    # 1. Load datasets
    train_ds, val_ds, test_ds = load_datasets(batch_size=BATCH_SIZE)
    
    # 2. Create model
    print("🔨 Tạo model...")
    classifier = DogBreedClassifier()
    model = classifier.build(trainable_base=False)
    classifier.compile(learning_rate=LEARNING_RATE)
    
    print(f"\nModel parameters: {model.count_params():,}\n")
    
    # 3. Callbacks
    callbacks = create_callbacks()
    
    # 4. Training
    print("Bắt đầu training...\n")
    print("="*60)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        use_multiprocessing=False
    )
    
    print("\n" + "="*60)
    print("TRAINING HOÀN TẤT!")
    print("="*60 + "\n")
    
    # 5. Evaluate on test set
    print("📊 Đánh giá trên test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}\n")
    
    # 6. Save final model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = FINAL_MODEL_DIR / f'dog_breed_classifier_{timestamp}.keras'
    model.save(final_model_path)
    print(f"Model cuối cùng đã lưu tại: {final_model_path}\n")
    
    # 7. Save class names
    save_class_names()
    
    # 8. Save model info
    metadata = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'epochs_trained': int(len(history.history['loss'])),
        'batch_size': int(BATCH_SIZE),
        'learning_rate': float(LEARNING_RATE)
    }
    save_model_info(model, FINAL_MODEL_DIR, metadata)
    
    # 9. Save training history
    history_path = LOGS_DIR / f'history_{timestamp}.json'
    save_training_history(history, history_path)
    
    # 10. Export to TFLite
    print("\nExport sang TFLite...")
    tflite_path = FINAL_MODEL_DIR / f'dog_breed_classifier_{timestamp}.tflite'
    export_to_tflite(model, tflite_path, quantize=True)
    
    print("\n" + "="*60)
    print("TẤT CẢ HOÀN TẤT!")
    print("="*60)
    print(f"\nModels tại: {FINAL_MODEL_DIR}")
    print(f"Logs tại: {LOGS_DIR}\n")
    
    return model, history


if __name__ == '__main__':
    # Setup GPU (nếu có)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Phát hiện {len(gpus)} GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("Không phát hiện GPU, dùng CPU")
    
    # Train
    model, history = train_model()
