"""
Script chia dữ liệu thành train/val/test splits
"""
import os
import shutil
from pathlib import Path
import json
import random
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

# Seed cho reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Cấu hình
RAW_DATA_DIR = Path('data/raw')
PROCESSED_DATA_DIR = Path('data/processed')
TRAIN_RATIO = 0.70  # 70%
VAL_RATIO = 0.15    # 15%
TEST_RATIO = 0.15   # 15%

# Danh sách 10 giống chó
BREED_NAMES = [
    'Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd',
    'Golden_Retriever', 'Labrador_Retriever', 'Poodle',
    'Rottweiler', 'Yorkshire_Terrier'
]


def check_data_integrity():
    """Kiểm tra tính toàn vẹn của dữ liệu gốc"""
    print("="*60)
    print("KIỂM TRA DỮ LIỆU GỐC")
    print("="*60 + "\n")
    
    total_images = 0
    breed_counts = {}
    
    for breed in BREED_NAMES:
        breed_path = RAW_DATA_DIR / breed
        
        if not breed_path.exists():
            print(f"❌ Thiếu thư mục: {breed}")
            continue
        
        # Đếm số ảnh (jpg, jpeg, png)
        images = list(breed_path.glob('*.jpg')) + \
                 list(breed_path.glob('*.jpeg')) + \
                 list(breed_path.glob('*.png')) + \
                 list(breed_path.glob('*.JPG')) + \
                 list(breed_path.glob('*.JPEG'))
        
        count = len(images)
        breed_counts[breed] = count
        total_images += count
        
        status = "✅" if count == 100 else "⚠️"
        print(f"{status} {breed:25s}: {count:3d} ảnh")
    
    print(f"\n{'='*60}")
    print(f"TỔNG CỘNG: {total_images} ảnh")
    print(f"TRUNG BÌNH: {total_images/len(BREED_NAMES):.1f} ảnh/giống")
    print(f"{'='*60}\n")
    
    return breed_counts, total_images


def get_all_images_with_labels():
    """Lấy tất cả đường dẫn ảnh và nhãn"""
    all_images = []
    all_labels = []
    
    for label_idx, breed in enumerate(BREED_NAMES):
        breed_path = RAW_DATA_DIR / breed
        
        # Lấy tất cả ảnh
        images = list(breed_path.glob('*.jpg')) + \
                 list(breed_path.glob('*.jpeg')) + \
                 list(breed_path.glob('*.png')) + \
                 list(breed_path.glob('*.JPG')) + \
                 list(breed_path.glob('*.JPEG'))
        
        for img_path in images:
            all_images.append(str(img_path))
            all_labels.append(label_idx)
    
    return np.array(all_images), np.array(all_labels)


def split_data(all_images, all_labels):
    """
    Chia dữ liệu thành train/val/test với stratification
    
    Returns:
        Dict chứa paths cho train, val, test
    """
    print("="*60)
    print("CHIA DỮ LIỆU")
    print("="*60 + "\n")
    
    # Chia train vs (val+test)
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images, all_labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=all_labels,
        random_state=RANDOM_SEED
    )
    
    # Chia val vs test
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_labels,
        random_state=RANDOM_SEED
    )
    
    # In thống kê
    print(f"📦 Train set: {len(train_imgs)} ảnh ({len(train_imgs)/len(all_images)*100:.1f}%)")
    print(f"📦 Val set:   {len(val_imgs)} ảnh ({len(val_imgs)/len(all_images)*100:.1f}%)")
    print(f"📦 Test set:  {len(test_imgs)} ảnh ({len(test_imgs)/len(all_images)*100:.1f}%)")
    
    # Kiểm tra phân bố theo lớp
    print(f"\n{'Breed':<25} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("-"*60)
    
    for idx, breed in enumerate(BREED_NAMES):
        train_count = np.sum(train_labels == idx)
        val_count = np.sum(val_labels == idx)
        test_count = np.sum(test_labels == idx)
        print(f"{breed:<25} {train_count:<8} {val_count:<8} {test_count:<8}")
    
    return {
        'train': {'images': train_imgs, 'labels': train_labels},
        'val': {'images': val_imgs, 'labels': val_labels},
        'test': {'images': test_imgs, 'labels': test_labels}
    }


def copy_images_to_splits(splits_data):
    """Copy ảnh từ raw sang thư mục train/val/test"""
    print(f"\n{'='*60}")
    print("COPY ẢNH VÀO CÁC SPLITS")
    print("="*60 + "\n")
    
    # Xóa thư mục processed cũ nếu có
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)
        print(f"🗑️  Đã xóa thư mục processed cũ\n")
    
    # Tạo cấu trúc thư mục
    for split in ['train', 'val', 'test']:
        for breed in BREED_NAMES:
            split_dir = PROCESSED_DATA_DIR / split / breed
            split_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy ảnh
    for split_name, split_data in splits_data.items():
        print(f"📁 Đang copy {split_name} set...")
        
        for img_path, label in zip(split_data['images'], split_data['labels']):
            breed_name = BREED_NAMES[label]
            src_path = Path(img_path)
            dst_path = PROCESSED_DATA_DIR / split_name / breed_name / src_path.name
            
            shutil.copy2(src_path, dst_path)
        
        print(f"   ✅ Đã copy {len(split_data['images'])} ảnh")
    
    print(f"\n✅ Copy hoàn tất!\n")


def save_split_info(splits_data):
    """Lưu thông tin về splits để reproducibility"""
    split_info = {
        'random_seed': RANDOM_SEED,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'breed_names': BREED_NAMES,
        'splits': {}
    }
    
    for split_name, split_data in splits_data.items():
        split_info['splits'][split_name] = {
            'count': len(split_data['images']),
            'images': [str(Path(p).relative_to(RAW_DATA_DIR)) for p in split_data['images']],
            'labels': split_data['labels'].tolist()
        }
    
    save_path = Path('data/data_splits.json')
    with open(save_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"💾 Đã lưu thông tin splits tại: {save_path}")


def verify_processed_data():
    """Xác minh dữ liệu đã xử lý"""
    print(f"\n{'='*60}")
    print("XÁC MINH DỮ LIỆU ĐÃ XỬ LÝ")
    print("="*60 + "\n")
    
    for split in ['train', 'val', 'test']:
        split_dir = PROCESSED_DATA_DIR / split
        total = 0
        
        print(f"📂 {split.upper()} set:")
        for breed in BREED_NAMES:
            breed_dir = split_dir / breed
            count = len(list(breed_dir.glob('*.jpg')) + 
                       list(breed_dir.glob('*.jpeg')) + 
                       list(breed_dir.glob('*.png')))
            total += count
            print(f"   {breed:<25}: {count} ảnh")
        
        print(f"   {'TỔNG':<25}: {total} ảnh\n")


def main():
    """Main function"""
    print("\n🐕 DOG BREED DATA PREPARATION\n")
    
    # 1. Kiểm tra dữ liệu gốc
    breed_counts, total = check_data_integrity()
    
    if total == 0:
        print("❌ Không tìm thấy dữ liệu! Kiểm tra lại thư mục data/raw/")
        return
    
    # 2. Lấy tất cả ảnh và nhãn
    print("📥 Đang load danh sách ảnh...")
    all_images, all_labels = get_all_images_with_labels()
    print(f"✅ Đã load {len(all_images)} ảnh\n")
    
    # 3. Chia dữ liệu
    splits_data = split_data(all_images, all_labels)
    
    # 4. Copy ảnh vào các splits
    copy_images_to_splits(splits_data)
    
    # 5. Lưu thông tin splits
    save_split_info(splits_data)
    
    # 6. Xác minh
    verify_processed_data()
    
    print("="*60)
    print("✅ DATA PREPARATION HOÀN TẤT!")
    print("="*60)
    print(f"\n📁 Dữ liệu đã xử lý tại: {PROCESSED_DATA_DIR}")
    print(f"📄 Thông tin splits tại: data/data_splits.json\n")


if __name__ == '__main__':
    main()
