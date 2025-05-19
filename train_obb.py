from ultralytics import YOLO

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

device = [2]

# Load a model
model = YOLO("yolo12x-obb.yaml")  # build a new model from YAML
# model = YOLO("yolo12-obbx.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo12x.yaml").load("yolo12x.pt")  # build from YAML and transfer weights

project = "results/ultralytics/yolov12/xobb"

# Train the model
results = model.train(data="ultralytics/cfg/datasets/CODrone.yaml", epochs=200, device=device, project=project, batch=16, optimizer='SGD',  name='codrone_vml4_')