"""
Định nghĩa kiến trúc model
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import NUM_CLASSES, IMG_SIZE, DROPOUT_RATE, BACKBONE


class DogBreedClassifier:
    """Wrapper class cho model phân loại giống chó"""
    
    def __init__(self, num_classes=NUM_CLASSES, img_size=IMG_SIZE, backbone=BACKBONE):
        self.num_classes = num_classes
        self.img_size = img_size
        self.backbone = backbone
        self.model = None
        
    def build(self, trainable_base=False):
        """Xây dựng model với pretrained backbone"""
        
        # Load backbone
        if self.backbone == 'efficientnet_b3':
            base_model = tf.keras.applications.EfficientNetB3(
                include_top=False,
                weights='imagenet',
                input_shape=(self.img_size, self.img_size, 3)
            )
        elif self.backbone == 'efficientnet_b4':
            base_model = tf.keras.applications.EfficientNetB4(
                include_top=False,
                weights='imagenet',
                input_shape=(self.img_size, self.img_size, 3)
            )
        elif self.backbone == 'resnet50':
            base_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=(self.img_size, self.img_size, 3)
            )
        else:
            raise ValueError(f"Backbone {self.backbone} không được hỗ trợ")
        
        # Đóng băng backbone
        base_model.trainable = trainable_base
        
        # Xây dựng model
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs, name=f'dog_breed_{self.backbone}')
        
        return self.model
    
    def compile(self, learning_rate=1e-3):
        """Compile model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            jit_compile=False
        )
        
    def summary(self):
        """In ra summary của model"""
        return self.model.summary()
    
    def save(self, filepath):
        """Lưu model"""
        self.model.save(filepath)
        print(f"Model đã lưu tại: {filepath}")
        
    def load(self, filepath):
        """Load model từ file"""
        self.model = keras.models.load_model(filepath)
        print(f"Model đã load từ: {filepath}")
        return self.model


if __name__ == '__main__':
    # Test model
    print("Tạo model...")
    classifier = DogBreedClassifier()
    model = classifier.build()
    classifier.compile()
    
    print("\nModel summary:")
    classifier.summary()
    
    print(f"\nModel parameters: {model.count_params():,}")
