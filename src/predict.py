"""
Script inference trên ảnh mới
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import tensorflow as tf
import numpy as np
from PIL import Image

from config import IMG_SIZE
from utils import load_class_names


class DogBreedPredictor:
    """Class để inference model phân loại giống chó"""
    
    def __init__(self, model_path, class_names_path=None):
        """
        Args:
            model_path: Đường dẫn đến file .keras
            class_names_path: Đường dẫn đến file class_names.json
        """
        self.model_path = Path(model_path)
        
        # Load model
        print(f"Loading model từ: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded!")
        
        # Load class names
        self.class_names = load_class_names(class_names_path)
        
    def preprocess_image(self, image_path):
        """Tiền xử lý ảnh"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32)
        
        # Preprocess theo EfficientNet
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path, top_k=3):
        """
        Dự đoán giống chó từ ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            top_k: Số lượng dự đoán top trả về
            
        Returns:
            List of tuples (breed_name, confidence)
        """
        # Preprocess
        img_array = self.preprocess_image(image_path)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Top-K results
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        results = [
            (self.class_names[idx], float(predictions[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def predict_and_display(self, image_path, top_k=3):
        """Dự đoán và in kết quả"""
        results = self.predict(image_path, top_k)
        
        print(f"\nDự đoán cho ảnh: {Path(image_path).name}")
        print("="*60)
        print(f"\nKết quả: {results[0][0]} ({results[0][1]*100:.2f}%)")
        
        print(f"\nTop-{top_k} dự đoán:")
        for i, (breed, conf) in enumerate(results, 1):
            bar = '█' * int(conf * 40)
            print(f"{i}. {breed:25s} {bar} {conf*100:.2f}%")
        
        print()
        
        return results


def main():
    """Demo inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dự đoán giống chó')
    parser.add_argument('--model', type=str, required=True, help='Đường dẫn model (.keras)')
    parser.add_argument('--image', type=str, required=True, help='Đường dẫn ảnh')
    parser.add_argument('--top-k', type=int, default=3, help='Số top predictions')
    
    args = parser.parse_args()
    
    # Tạo predictor
    predictor = DogBreedPredictor(args.model)
    
    # Predict
    predictor.predict_and_display(args.image, args.top_k)


if __name__ == '__main__':
    main()
