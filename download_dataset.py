"""
Dataset Setup Script
Downloads the Brain MRI Images dataset from Kaggle and organizes it
into the required dataset/train and dataset/test structure.

Prerequisites:
  pip install kagglehub

Usage:
  python download_dataset.py
"""

import os
import shutil
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
SPLIT_RATIO = 0.8  # 80% train, 20% test


def download_dataset():
    """Download dataset using kagglehub."""
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        os.system("pip install kagglehub")
        import kagglehub

    print("Downloading Brain MRI dataset from Kaggle...")
    path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
    print(f"Downloaded to: {path}")
    return path


def organize_dataset(source_path):
    """Split images into train/test with tumor/normal subfolders."""
    # Find the actual image folders (yes/no mapping to tumor/normal)
    # The Kaggle dataset uses 'yes' for tumor and 'no' for normal
    source_tumor = None
    source_normal = None

    for root, dirs, files in os.walk(source_path):
        basename = os.path.basename(root).lower()
        if basename == "yes":
            source_tumor = root
        elif basename == "no":
            source_normal = root

    if not source_tumor or not source_normal:
        # Try 'tumor' and 'normal' folder names
        for root, dirs, files in os.walk(source_path):
            basename = os.path.basename(root).lower()
            if basename == "tumor":
                source_tumor = root
            elif basename in ("normal", "healthy"):
                source_normal = root

    if not source_tumor or not source_normal:
        print("ERROR: Could not find tumor/normal (or yes/no) folders in the dataset.")
        print(f"Please manually place images into:")
        print(f"  {os.path.join(TRAIN_DIR, 'tumor')}")
        print(f"  {os.path.join(TRAIN_DIR, 'normal')}")
        print(f"  {os.path.join(TEST_DIR, 'tumor')}")
        print(f"  {os.path.join(TEST_DIR, 'normal')}")
        return

    # Create directory structure
    for split in [TRAIN_DIR, TEST_DIR]:
        for cls in ["tumor", "normal"]:
            os.makedirs(os.path.join(split, cls), exist_ok=True)

    # Split and copy images
    for src_folder, cls_name in [(source_tumor, "tumor"), (source_normal, "normal")]:
        images = [f for f in os.listdir(src_folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]
        random.shuffle(images)
        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        for img in train_imgs:
            shutil.copy2(os.path.join(src_folder, img),
                         os.path.join(TRAIN_DIR, cls_name, img))
        for img in test_imgs:
            shutil.copy2(os.path.join(src_folder, img),
                         os.path.join(TEST_DIR, cls_name, img))

        print(f"  {cls_name}: {len(train_imgs)} train, {len(test_imgs)} test")


if __name__ == "__main__":
    random.seed(42)

    if os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR):
        print("Dataset directory already exists. Skipping download.")
    else:
        source = download_dataset()
        print("\nOrganizing into train/test splits...")
        organize_dataset(source)

    print("\nDataset ready!")
    print(f"  Train: {TRAIN_DIR}")
    print(f"  Test:  {TEST_DIR}")
