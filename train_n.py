from ultralytics import YOLO
from datetime import datetime
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

device = [3]

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

project = "results/ultralytics/yolov11/n/car"
date = datetime.now().strftime("%Y%m%d_%H%M")
name = f'{date}'

# Train the model
results = model.train(data="ultralytics/cfg/datasets/_carclass.yaml", epochs=200, imgsz=640, device=device, project=project, batch=12, optimizer='SGD',  name=name, pretrained=True)