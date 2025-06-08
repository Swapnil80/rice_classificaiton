import os
import shutil
from pathlib import Path

def setup_dataset():
    # Create rice_dataset directory if it doesn't exist
    os.makedirs('rice_dataset', exist_ok=True)
    
    # Source and destination directories
    src_dir = 'Rice_Image_Dataset'
    dst_dir = 'rice_dataset'
    
    # Rice varieties
    varieties = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    
    # Copy each variety's images
    for variety in varieties:
        src_variety_dir = os.path.join(src_dir, variety)
        dst_variety_dir = os.path.join(dst_dir, variety)
        
        # Create destination directory
        os.makedirs(dst_variety_dir, exist_ok=True)
        
        # Copy all images
        for img_file in os.listdir(src_variety_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(src_variety_dir, img_file)
                dst_path = os.path.join(dst_variety_dir, img_file)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {img_file} to {dst_variety_dir}")

if __name__ == "__main__":
    setup_dataset() 