from ultralytics import YOLO

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

device = [1]

# Load a model
model = YOLO("yolo12x.yaml")  # build a new model from YAML
model = YOLO("yolo12x.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo12x.yaml").load("yolo12x.pt")  # build from YAML and transfer weights

project = "results/ultralytics/yolov12/x"

# Train the model 1
results = model.train(data="ultralytics/cfg/datasets/VisDrone.yaml", epochs=300, imgsz=640, device=device, project=project, batch=12, optimizer='SGD',  name='visdrone_vml6_', pretrained=False)