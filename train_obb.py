from ultralytics import YOLO

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

device = [2]

# Load a model
model = YOLO("yolo11x-obb.yaml")  # build a new model from YAML
model = YOLO("yolo11x-obb.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x-obb.yaml").load("yolo11x-obb.pt")  # build from YAML and transfer weights

project = "results/ultralytics/yolov11/xobb"

# Train the model
results = model.train(data="ultralytics/cfg/datasets/CODrone.yaml", epochs=300, device=device, project=project, batch=12, optimizer='SGD',  name='codrone_vml6_')