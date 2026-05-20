import os
import shutil
import random

base = "E:/HumanSyntheticProject/dataset"
images = os.path.join(base, "images")
labels = os.path.join(base, "labels")

train_img = os.path.join(base, "train/images")
train_lbl = os.path.join(base, "train/labels")
val_img   = os.path.join(base, "val/images")
val_lbl   = os.path.join(base, "val/labels")

os.makedirs(train_img, exist_ok=True)
os.makedirs(train_lbl, exist_ok=True)
os.makedirs(val_img,   exist_ok=True)
os.makedirs(val_lbl,   exist_ok=True)

files = [f.replace(".png", "") for f in os.listdir(images) if f.endswith(".png")]
random.shuffle(files)

split = int(len(files) * 0.8)
train_files = files[:split]
val_files   = files[split:]

for f in train_files:
    shutil.copy(os.path.join(images, f+".png"), os.path.join(train_img, f+".png"))
    shutil.copy(os.path.join(labels, f+".txt"), os.path.join(train_lbl, f+".txt"))

for f in val_files:
    shutil.copy(os.path.join(images, f+".png"), os.path.join(val_img, f+".png"))
    shutil.copy(os.path.join(labels, f+".txt"), os.path.join(val_lbl, f+".txt"))

print(f"Train: {len(train_files)} images")
print(f"Val:   {len(val_files)} images")
print("Done!")