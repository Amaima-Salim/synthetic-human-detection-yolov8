# Human Synthetic Dataset Project

A computer vision research project that explores whether a model trained **entirely on synthetic data** can detect humans in real-world photographs. The project investigates the sim-to-real domain gap using Blender-generated images and YOLOv8.

---

## Project Overview

The core research question is: *Can a YOLOv8 object detection model trained only on synthetic Blender renders detect real humans in photographs?*

This project builds a complete automated pipeline — from 3D scene generation in Blender to model training and real-world evaluation — without using any real labeled images for training.

---

## Pipeline

```
Blender Scene → Python Script → Synthetic Dataset → YOLOv8 Training → Real Image Detection
```

1. **Scene Setup** — 6 human poses imported from Mixamo, 33 HDRIs from Polyhaven
2. **Dataset Generation** — Python script randomizes pose, position, lighting, camera angle per render
3. **Auto Labeling** — Bounding boxes computed via Blender camera projection math (no manual labeling)
4. **Training** — YOLOv8n trained entirely on synthetic images
5. **Evaluation** — Model tested on real photographs from Pexels

---

## Dataset Details

| Property | Value |
|---|---|
| Total images | 1500 |
| Image resolution | 640×640 |
| Annotation format | YOLO (.txt) |
| Human poses | 6 (idle, walking, running, texting, looking around, T-pose) |
| HDRI backgrounds | 33 (indoor, outdoor, urban) |
| Humans per image | 1, 2, or 3 (randomized) |
| Labels | Auto-generated via camera projection |

---

## Results

| Attempt | Dataset | Epochs | Confidence | Detections |
|---|---|---|---|---|
| v1 | 500 images, T-pose only, 10 HDRIs | 50 | 0.3 | 2/23 (8.7%) |
| v1 | Same | 50 | 0.1 | 5/23 (21.7%) |
| v2 | 1500 images, 6 poses, 33 HDRIs | 100 | 0.15 | 5/23 (21.7%) |

### Key Findings

- Detection rate remained consistent at **21.7%** even after significantly expanding the dataset
- Model successfully detected humans on **plain and simple backgrounds**
- Model struggled with **complex scenes, crowds, and heavily clothed subjects**
- Increasing dataset size and pose variety did not improve results beyond the initial threshold
- The primary bottleneck is the **visual appearance gap** between a textureless 3D mannequin and real clothed humans

---

## Conclusion

Synthetic data alone can partially bridge the sim-to-real gap for human detection. The results demonstrate that **dataset size and pose variety are not the limiting factors** — the critical bottleneck is photorealistic appearance. A textureless mannequin model fundamentally differs from real humans in clothing, skin texture, and hair, which limits detection accuracy regardless of dataset size.

Future improvements would require:
- Photorealistic textured human models
- Clothing variation per render
- Mixing a small number of real labeled images into training data

---

## Project Structure

```
HumanSyntheticProject/
├── blender/
│   └── scene.blend              # Blender scene with all poses and camera
├── dataset/
│   ├── images/                  # Generated synthetic images
│   ├── labels/                  # Auto-generated YOLO labels
│   ├── train/                   # Training split (80%)
│   ├── val/                     # Validation split (20%)
│   └── dataset.yaml             # YOLO dataset config
├── hdris/                       # HDRI environment maps (not included)
├── poses/                       # Mixamo FBX pose files (not included)
├── scripts/
│   └── split_dataset.py         # Train/val split script
├── test_images/                 # Real photographs for evaluation
├── training/
│   ├── train.py                 # YOLOv8 training script
│   ├── predict.py               # Inference script
│   ├── yolov8n.pt               # Base YOLOv8 model
│   └── runs/                    # Training outputs and weights
└── results/                     # Prediction results on real images
```

---

## Setup and Usage

### Requirements

```
Python 3.10+
Blender 5.1+
ultralytics
torch
```

### Install dependencies

```bash
pip install ultralytics
```

### Generate dataset

1. Open `blender/scene.blend` in Blender
2. Go to the Scripting tab
3. Paste and run the generation script
4. Images and labels save to `dataset/images/` and `dataset/labels/`

### Split dataset

```bash
python scripts/split_dataset.py
```

### Train model

```bash
python training/train.py
```

### Run predictions on real images

Place test images in `test_images/` then:

```bash
python training/predict.py
```

Results save to `results/`.

---

## Tools and Libraries

- **Blender 5.1** — 3D scene creation and rendering
- **Mixamo** — Human pose FBX files
- **Polyhaven** — HDRI environment maps
- **YOLOv8n** — Object detection model
- **Ultralytics** — Training and inference framework
- **Python** — Dataset generation and automation scripts

---

## Notes

- Training was performed on CPU (Intel Core i5-8365U) — no GPU used
- HDRI files and pose FBX files are not included in the repository due to licensing
- Test images sourced from Pexels (free license)