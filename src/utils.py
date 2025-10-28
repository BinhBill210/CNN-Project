"""
Các hàm tiện ích (đã sửa lỗi JSON serialization)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from config import BREED_NAMES, get_class_names_path


def convert_to_serializable(obj):
    """
    Chuyển đổi object sang dạng có thể serialize JSON
    Xử lý TensorFlow tensors, NumPy arrays, etc.
    """
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (tf.Tensor, tf.Variable)):
        return obj.numpy().tolist() if hasattr(obj, 'numpy') else str(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_class_names(save_path=None):
    """Lưu danh sách tên các lớp"""
    if save_path is None:
        save_path = get_class_names_path()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({'classes': BREED_NAMES}, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Class names đã lưu tại: {save_path}")


def load_class_names(load_path=None):
    """Load danh sách tên các lớp"""
    if load_path is None:
        load_path = get_class_names_path()
    
    with open(load_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['classes']


def save_model_info(model, save_dir, metadata=None):
    """
    Lưu thông tin về model (đã sửa lỗi serialization)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Tính toán params an toàn
    try:
        total_params = int(model.count_params())
    except:
        total_params = 0
    
    try:
        trainable_params = int(sum([tf.size(w).numpy() for w in model.trainable_weights]))
    except:
        trainable_params = 0
    
    info = {
        'model_name': str(model.name),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'timestamp': datetime.now().isoformat()
    }
    
    # Thêm metadata và convert sang serializable
    if metadata:
        # Convert metadata sang dạng serializable
        metadata_clean = convert_to_serializable(metadata)
        info.update(metadata_clean)
    
    info_path = save_dir / 'model_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Model info đã lưu tại: {info_path}")


def export_to_tflite(model, save_path, quantize=False):
    """Export model sang TFLite format"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("⚙️  Đang quantize model...")
    
    tflite_model = converter.convert()
    
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ TFLite model đã lưu tại: {save_path}")
    print(f"📦 Kích thước: {file_size:.2f} MB")


def save_training_history(history, save_path):
    """
    Lưu training history (đã sửa lỗi serialization)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert history thành dict với giá trị Python native
    history_dict = {}
    for key, values in history.history.items():
        # Convert mỗi value sang float Python thuần
        history_dict[key] = [float(val) for val in values]
    
    # Lưu JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(history_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Training history đã lưu tại: {save_path}")


if __name__ == '_main_':
    # Test utilities
    print("🧪 Testing utilities...")
    
    # Test convert function
    test_data = {
        'numpy_int': np.int32(42),
        'numpy_float': np.float32(3.14),
        'numpy_array': np.array([1, 2, 3]),
        'normal_data': {'a': 1, 'b': 2}
    }
    
    converted = convert_to_serializable(test_data)
    print(f"\n✅ Converted data: {converted}")
    
    # Test saving class names
    save_class_names()
    classes = load_class_names()
    print(f"\n📋 Classes: {classes}")