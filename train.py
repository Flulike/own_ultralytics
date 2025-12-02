from ultralytics import YOLO
from datetime import datetime
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

device = [2,3]

# Load a model
model = YOLO("yolo11s.yaml")  # build a new model from YAML
model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11s.yaml").load("yolo11s.pt")  # build from YAML and transfer weights

project = "results/ultralytics/s"
date = datetime.now().strftime("%Y%m%d_%H%M")
name = f'{date}_ggmix'

# Train the model
results = model.train(data="ultralytics/cfg/datasets/VisDrone.yaml", epochs=300, imgsz=640, device=device, project=project, batch=12, optimizer='SGD',  name=name, pretrained=True)