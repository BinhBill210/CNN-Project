"""
DataLoader và augmentation pipeline
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import (
    IMG_SIZE, BATCH_SIZE, BREED_NAMES,
    get_train_dir, get_val_dir, get_test_dir
)

AUTOTUNE = tf.data.AUTOTUNE


def get_data_augmentation():
    """Tạo augmentation pipeline cho training"""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="data_augmentation")


def preprocess_image(image, label, is_training=False, augmentation=None):
    """Tiền xử lý ảnh cho EfficientNet"""
    # Resize
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # Augmentation (chỉ khi training)
    if is_training and augmentation is not None:
        image = augmentation(image, training=True)
    
    # Preprocess theo EfficientNet
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    
    return image, label


def create_dataset_from_directory(data_dir, is_training=False, batch_size=BATCH_SIZE):
    """
    Tạo tf.data.Dataset từ thư mục
    
    Args:
        data_dir: Path object hoặc string đến train/val/test
        is_training: True nếu là training set
        batch_size: Batch size
        
    Returns:
        tf.data.Dataset
    """
    data_dir = Path(data_dir)
    
    print(f"   Loading từ: {data_dir.relative_to(Path.cwd())}")
    
    # Load dataset
    dataset = keras.utils.image_dataset_from_directory(
        str(data_dir),
        labels='inferred',
        label_mode='int',
        class_names=BREED_NAMES,
        color_mode='rgb',
        batch_size=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=is_training,
        seed=42
    )
    
    # Augmentation
    if is_training:
        augmentation = get_data_augmentation()
        dataset = dataset.map(
            lambda x, y: preprocess_image(x, y, is_training=True, augmentation=augmentation),
            num_parallel_calls=AUTOTUNE
        )
    else:
        dataset = dataset.map(
            lambda x, y: preprocess_image(x, y, is_training=False),
            num_parallel_calls=AUTOTUNE
        )
    
    # Shuffle, batch, prefetch
    if is_training:
        dataset = dataset.shuffle(1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset


def load_datasets(batch_size=BATCH_SIZE):
    """
    Load tất cả train/val/test datasets
    
    Returns:
        tuple: (train_ds, val_ds, test_ds)
    """
    print("📥 Loading datasets...")
    
    train_ds = create_dataset_from_directory(
        get_train_dir(),
        is_training=True,
        batch_size=batch_size
    )
    
    val_ds = create_dataset_from_directory(
        get_val_dir(),
        is_training=False,
        batch_size=batch_size
    )
    
    test_ds = create_dataset_from_directory(
        get_test_dir(),
        is_training=False,
        batch_size=batch_size
    )
    
    print("✅ Datasets loaded!\n")
    
    return train_ds, val_ds, test_ds


def get_dataset_info(dataset, name='Dataset'):
    """In thông tin về dataset"""
    for images, labels in dataset.take(1):
        print(f"📊 {name}:")
        print(f"   Image batch shape: {images.shape}")
        print(f"   Label batch shape: {labels.shape}")
        print(f"   Image dtype: {images.dtype}")
        print(f"   Label dtype: {labels.dtype}")
        print()


if __name__ == '__main__':
    print(f"📁 Thư mục làm việc: {Path.cwd()}\n")
    
    train_ds, val_ds, test_ds = load_datasets()
    
    get_dataset_info(train_ds, 'Train')
    get_dataset_info(val_ds, 'Validation')
    get_dataset_info(test_ds, 'Test')
