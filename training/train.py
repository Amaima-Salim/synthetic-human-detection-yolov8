from ultralytics import YOLO

model = YOLO("E:/HumanSyntheticProject/training/yolov8n.pt")

model.train(
    data="E:/HumanSyntheticProject/dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=2,
    cache=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    degrees=15,
    translate=0.1,
    scale=0.5,
    project="E:/HumanSyntheticProject/training/runs/detect",
    name="train_v3"
)