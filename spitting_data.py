import random
import shutil
from pathlib import Path

# --- CONFIG ---
dataset_dir = Path("C:\\Users\\Admin\\Uni\\LatexOCR")
images_dir = dataset_dir / "preprocessed_images"
labels_file = dataset_dir / "CROHME_math.txt"
output_dir = Path("C:\\Users\\Admin\\Uni\\LatexOCR\\splitted_dataset")
train_ratio, val_ratio, test_ratio = 0.90, 0.05, 0.05
seed = 42
# -------------

random.seed(seed)

# 1) Load all labels
with open(labels_file, 'r', encoding='utf-8') as f:
    all_labels = [line.strip() for line in f.readlines()]

# 2) Get list of available images and their corresponding labels
image_paths = list(images_dir.glob("*.png"))
image_data = []

for img_path in image_paths:
    if img_path.stem.isdigit():
        idx = int(img_path.stem)
        if idx < len(all_labels) and all_labels[idx].strip():  # Check if label is not empty
            image_data.append({
                'path': img_path,
                'label': all_labels[idx].strip(),  # Clean up whitespace
                'idx': idx
            })

# 3) Shuffle the data while keeping images and labels paired
random.shuffle(image_data)

# 4) Calculate split sizes
n_total = len(image_data)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

# 5) Split the data
train_data = image_data[:n_train]
val_data = image_data[n_train:n_train + n_val]
test_data = image_data[n_train + n_val:]

splits = {
    "train": train_data,
    "val": val_data,
    "test": test_data
}

# 6) Create split folders and copy files + write labels
for split_name, data in splits.items():
    split_img_dir = output_dir / split_name / "images"
    split_img_dir.mkdir(parents=True, exist_ok=True)
    split_labels_path = output_dir / split_name / "labels.txt"

    # Sort by original index for consistent ordering
    data_sorted = sorted(data, key=lambda x: x['idx'])

    # Write labels and copy images
    with open(split_labels_path, "w", encoding="utf-8") as f:
        for item in data_sorted:
            # Copy image
            dst_img = split_img_dir / item['path'].name
            shutil.copy2(item['path'], dst_img)
            # Write corresponding label (only the label text)
            f.write(f"{item['label']}\n")

    print(f"{split_name}: {len(data)} images, labels saved to {split_labels_path}")

print("✅ Done splitting!")