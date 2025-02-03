from collections import defaultdict
import os
import random
from PIL import Image, ImageOps
import albumentations as A
import numpy as np

# Path to the organized data directory
base_dir = "organized_data"  # Adjust this if needed

# Initialize dictionaries to store the counts
class_counts = defaultdict(lambda: {'train': 0, 'validation': 0, 'test': 0})

# Iterate over each split (train, validation, test)
for split in ['train', 'validation', 'test']:
    split_dir = os.path.join(base_dir, split)
    
    # Iterate over each class in the split directory
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        
        # Count the number of images in each class
        if os.path.isdir(class_dir):
            num_images = len(os.listdir(class_dir))
            class_counts[class_name][split] = num_images

# Display the class counts
print("Class Counts by Split:")
for class_name, counts in class_counts.items():
    print(f"{class_name}: {counts}")

# Calculate total images per class
print("\nTotal Images per Class:")
for class_name, counts in class_counts.items():
    total = sum(counts.values())
    print(f"{class_name}: {total}")

# Define paths
base_dir = "organized_data"  # Adjust this if needed

# Desired counts for each split
train_target = 140
val_target = 30
test_target = 30

# Define augmentation pipeline
def get_augmentation_pipeline():
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1)
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RGBShift(p=0.5),
        A.CLAHE(clip_limit=4.0, p=0.2),
        A.RandomGamma(p=0.3),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3)
    ])

# Function to apply augmentation and save augmented images
def augment_and_save(image_path, save_dir, num_augments):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    aug_pipeline = get_augmentation_pipeline()
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(num_augments):
        augmented = aug_pipeline(image=image_np)
        augmented_image = Image.fromarray(augmented["image"])
        augmented_image.save(os.path.join(save_dir, f"{base_name}_aug_{i}.jpg"))

# Iterate over each split
for split, target_count in zip(['train', 'validation', 'test'], [train_target, val_target, test_target]):
    split_dir = os.path.join(base_dir, split)

    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        print(f"Processing {class_name} in {split} split.")
        
        # List valid image files
        images = [img for img in os.listdir(class_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
        current_count = len(images)

        if current_count < target_count:
            total_augments_needed = target_count - current_count
            
            if current_count > 0:
                num_augments_per_image = (total_augments_needed + current_count - 1) // current_count
            else:
                print(f"No images in {class_name} folder. Skipping.")
                continue

            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                try:
                    augment_and_save(img_path, class_dir, num_augments_per_image)
                except Exception as e:
                    print(f"Error augmenting {img_path}: {e}")

            # Adjust remainder
            remaining_augments = target_count - len(os.listdir(class_dir))
            if remaining_augments > 0:
                additional_images = random.sample(images, min(remaining_augments, len(images)))
                for img_name in additional_images:
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        augment_and_save(img_path, class_dir, 1)
                    except Exception as e:
                        print(f"Error augmenting {img_path}: {e}")

print("Augmentation completed for each split!")