"""
Script inference với xử lý lỗi robust
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from typing import List, Tuple, Optional

from config import IMG_SIZE, BREED_NAMES, get_class_names_path
from utils import load_class_names


class DogBreedPredictor:
    """Class inference với xử lý lỗi tốt"""
    
    def __init__(
        self, 
        model_path: str, 
        class_names_path: Optional[str] = None,
        img_size: int = IMG_SIZE
    ):
        """Khởi tạo với error handling"""
        self.model_path = Path(model_path).absolute()
        self.img_size = img_size
        
        # Kiểm tra file model
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model tại: {self.model_path}\n"
                f"Kiểm tra lại đường dẫn hoặc train model trước"
            )
        
        # Kiểm tra kích thước file
        file_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        print(f"Kích thước model: {file_size_mb:.2f} MB")
        
        if file_size_mb < 1:
            print(f"Cảnh báo: File model quá nhỏ ({file_size_mb:.2f} MB), có thể bị lỗi!")
        
        # Load model
        print(f"📥 Loading model từ: {self.model_path.name}")
        try:
            # Method 1: Load thông thường
            self.model = tf.keras.models.load_model(
                str(self.model_path),
                compile=True
            )
            print("Model loaded thành công!")
            
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            print("Thử method 2: Load without compile...")
            
            try:
                # Method 2: Load mà không compile
                self.model = tf.keras.models.load_model(
                    str(self.model_path),
                    compile=False
                )
                
                # Compile lại
                self.model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                print("✅ Model loaded thành công (method 2)!")
                
            except Exception as e2:
                print(f"\nCả 2 methods đều thất bại!")
                print(f"   Error 1: {e1}")
                print(f"   Error 2: {e2}")
                print(f"\nKhắc phục:")
                print(f"   1. Train lại model: python src/train.py")
                print(f"   2. Hoặc dùng checkpoint: models/checkpoints/best.keras")
                raise RuntimeError(f"Không thể load model")
        
        # Load class names
        try:
            if class_names_path:
                self.class_names = load_class_names(class_names_path)
            else:
                self.class_names = load_class_names(get_class_names_path())
            print(f"Loaded {len(self.class_names)} classes\n")
        except Exception as e:
            print(f"Không load được class names: {e}")
            print(f"   Dùng default classes từ config")
            self.class_names = BREED_NAMES
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess ảnh"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            img_array = np.array(img, dtype=np.float32)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            raise RuntimeError(f"Lỗi preprocess {image_path}: {e}")
    
    def predict(self, image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Dự đoán giống chó"""
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = [
            (self.class_names[idx], float(predictions[idx]))
            for idx in top_indices
        ]
        return results
    
    def predict_and_display(self, image_path: str, top_k: int = 3):
        """Dự đoán và hiển thị"""
        results = self.predict(image_path, top_k)
        
        print(f"\n{'='*60}")
        print(f"DỰ ĐOÁN GIỐNG CHÓ")
        print(f"{'='*60}")
        print(f"\nẢnh: {Path(image_path).name}")
        print(f"{'='*60}")
        
        top1_breed, top1_conf = results[0]
        print(f"\n Kết quả: {top1_breed.replace('_', ' ')}")
        print(f"   Độ tin cậy: {top1_conf*100:.2f}%")
        
        print(f"\nTop-{top_k} dự đoán:")
        print(f"{'-'*60}")
        
        for i, (breed, conf) in enumerate(results, 1):
            bar_length = 40
            filled = int(conf * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            breed_display = breed.replace('_', ' ')
            print(f"{i}. {breed_display:<25} {bar} {conf*100:5.2f}%")
        
        print(f"{'='*60}\n")
        return results


def main():
    parser = argparse.ArgumentParser(description='Dự đoán giống chó')
    parser.add_argument('--model', type=str, required=True, help='Path model .keras')
    parser.add_argument('--image', type=str, required=True, help='Path ảnh')
    parser.add_argument('--top-k', type=int, default=3, help='Top K predictions')
    parser.add_argument('--class-names', type=str, help='Path class_names.json (optional)')
    
    args = parser.parse_args()
    
    try:
        predictor = DogBreedPredictor(
            model_path=args.model,
            class_names_path=args.class_names
        )
        predictor.predict_and_display(args.image, args.top_k)
        
    except Exception as e:
        print(f"\nLỗi: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
