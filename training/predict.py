from ultralytics import YOLO
import os

model = YOLO("E:/HumanSyntheticProject/training/runs/detect/train_v2/weights/best.pt")

test_folder = "E:/HumanSyntheticProject/test_images"

model.predict(
    source=test_folder,
    save=True,
    conf=0.3,
    project="E:/HumanSyntheticProject/results",
    name="predict_v2"
)