from ultralytics import YOLO
from datetime import datetime
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

device = [0]

# Load a model
model = YOLO("yolo11x.yaml")  # build a new model from YAML
model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x.yaml").load("yolo11x.pt")  # build from YAML and transfer weights

project = "results/ultralytics/yolov11/x/abl"
date = datetime.now().strftime("%Y%m%d_%H%M")
name = f'{date}_down'

# Train the model
results = model.train(data="ultralytics/cfg/datasets/VisDrone.yaml", epochs=300, imgsz=640, device=device, project=project, batch=12, optimizer='SGD',  name=name, pretrained=False)