from ultralytics import YOLO

model = YOLO("E:/HumanSyntheticProject/training/yolov8n.pt")

model.train(
    data="E:/HumanSyntheticProject/dataset/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    project="E:/HumanSyntheticProject/training/runs/detect",
    name="train_v2"
)