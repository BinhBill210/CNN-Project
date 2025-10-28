"""
File cấu hình trung tâm cho toàn bộ project
Tất cả đường dẫn là tương đối từ thư mục gốc project
"""
from pathlib import Path

# ============================================================
# ĐƯỜNG DẪN (Tương đối từ thư mục gốc project)
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent  # Thư mục gốc project
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'
CHECKPOINT_DIR = MODEL_DIR / 'checkpoints'
FINAL_MODEL_DIR = MODEL_DIR / 'final'
RESULTS_DIR = PROJECT_ROOT / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots'
METRICS_DIR = RESULTS_DIR / 'metrics'
LOGS_DIR = RESULTS_DIR / 'logs'

# ============================================================
# DỮ LIỆU
# ============================================================
BREED_NAMES = [
    'Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd',
    'Golden_Retriever', 'Labrador_Retriever', 'Poodle',
    'Rottweiler', 'Yorkshire_Terrier'
]
NUM_CLASSES = len(BREED_NAMES)

# Tỷ lệ chia dữ liệu
TRAIN_RATIO = 0.70  # 70%
VAL_RATIO = 0.15    # 15%
TEST_RATIO = 0.15   # 15%

# ============================================================
# HYPERPARAMETERS
# ============================================================
IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
DROPOUT_RATE = 0.3

# Backbone model
BACKBONE = 'efficientnet_b3'  # 'efficientnet_b3', 'efficientnet_b4', 'resnet50'

# ============================================================
# TRAINING
# ============================================================
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# ============================================================
# PATHS HELPERS
# ============================================================
def get_train_dir():
    """Trả về đường dẫn thư mục train"""
    return PROCESSED_DATA_DIR / 'train'

def get_val_dir():
    """Trả về đường dẫn thư mục validation"""
    return PROCESSED_DATA_DIR / 'val'

def get_test_dir():
    """Trả về đường dẫn thư mục test"""
    return PROCESSED_DATA_DIR / 'test'

def get_data_splits_path():
    """Trả về đường dẫn file data_splits.json"""
    return DATA_DIR / 'data_splits.json'

def get_class_names_path():
    """Trả về đường dẫn file class_names.json"""
    return FINAL_MODEL_DIR / 'class_names.json'

def get_model_path(name='dog_breed_classifier'):
    """Trả về đường dẫn model cuối cùng"""
    return FINAL_MODEL_DIR / f'{name}.keras'

def get_tflite_path(name='dog_breed_classifier'):
    """Trả về đường dẫn TFLite model"""
    return FINAL_MODEL_DIR / f'{name}.tflite'

# ============================================================
# TẠO THƯ MỤC NẾU CHƯA TỒN TẠI
# ============================================================
def setup_directories():
    """Tạo tất cả thư mục cần thiết"""
    dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        CHECKPOINT_DIR,
        FINAL_MODEL_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        METRICS_DIR,
        LOGS_DIR
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    print("Đã tạo tất cả thư mục cần thiết")
    return dirs

if __name__ == '__main__':
    print("Cấu hình đường dẫn project:\n")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    
    print("\nTạo thư mục...")
    setup_directories()
