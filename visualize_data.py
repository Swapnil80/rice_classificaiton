import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict

def visualize_data(dataset_path='rice_dataset'):
    print(f"Analyzing dataset at: {dataset_path}")

    class_counts = defaultdict(int)
    sample_images = defaultdict(list)
    
    # Iterate through each class folder
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_counts[class_name] += 1
                    if len(sample_images[class_name]) < 3:  # Get up to 3 sample images per class
                        sample_images[class_name].append(os.path.join(class_path, img_name))

    # Plotting class distribution
    if class_counts:
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        print("\n--- Image counts per category ---")
        for class_name, count in sorted(class_counts.items()):
            print(f"{class_name}: {count} images")
        print("-----------------------------------")

        plt.figure(figsize=(10, 6))
        plt.bar(classes, counts, color='skyblue')
        plt.xlabel('Rice Variety')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Rice Varieties in Dataset')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("Class distribution plot displayed.")
    else:
        print("No image data found for class distribution analysis.")

    # Displaying sample images
    print("\nDisplaying sample images per class:")
    for class_name, img_paths in sample_images.items():
        if img_paths:
            plt.figure(figsize=(12, 4))
            plt.suptitle(f"Sample Images for {class_name}", fontsize=16)
            for i, img_path in enumerate(img_paths):
                plt.subplot(1, len(img_paths), i + 1)
                img = mpimg.imread(img_path)
                plt.imshow(img)
                plt.title(os.path.basename(img_path))
                plt.axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
        else:
            print(f"No sample images found for {class_name}.")

if __name__ == "__main__":
    visualize_data() 