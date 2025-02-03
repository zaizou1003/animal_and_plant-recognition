from datasets import load_dataset
import os
import shutil
import random
from collections import defaultdict
from PIL import Image
from io import BytesIO

# ds = load_dataset("mertcobanov/animals")
# ds = load_dataset("uran66/animals")
# ds = load_dataset("OttoYu/Tree-Species")
# Load the dataset
# Load the dataset
# Load dataset


datasets = [
    r"C:\Users\Mega-PC\.cache\huggingface\hub\datasets--mertcobanov--animals\snapshots\cfafe186d34b9cf24e232b58c3423aab055a917c\animals",
    r"C:\Users\Mega-PC\.cache\huggingface\hub\datasets--uran66--animals\snapshots\3adafe7c2233faa54d0df6a1d070bb1f1d3f5176\train",
]
# Directory where the organized data will be stored
base_dir = "organized_data"  # Adjust this if needed
os.makedirs(base_dir, exist_ok=True)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Function to create directory structure
def create_dir_structure(base_dir, class_name):
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(base_dir, split, class_name), exist_ok=True)
# Organize each dataset
for dataset in datasets:
    for class_name in os.listdir(dataset):
        class_path = os.path.join(dataset, class_name)
        
        # Skip non-directory files if any
        if not os.path.isdir(class_path):
            continue

        # Create directory structure for the class
        create_dir_structure(base_dir, class_name)

        # List all images in the class folder
        images = os.listdir(class_path)
        random.shuffle(images)

        # Calculate split indices
        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * (train_ratio + val_ratio))

        # Copy images to the respective folders
        for i, img_name in enumerate(images):
            src_path = os.path.join(class_path, img_name)
            
            if i < train_split:
                dst_path = os.path.join(base_dir, 'train', class_name, img_name)
            elif i < val_split:
                dst_path = os.path.join(base_dir, 'validation', class_name, img_name)
            else:
                dst_path = os.path.join(base_dir, 'test', class_name, img_name)
            
            shutil.copy(src_path, dst_path)

print("Data organized successfully!")