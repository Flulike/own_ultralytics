from ultralytics import YOLO

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

device = [0, 1]

# Load a model
model = YOLO("yolo11x.yaml")  # build a new model from YAML
model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x.yaml").load("yolo11x.pt")  # build from YAML and transfer weights

project = "results/ultralytics/yolov11/x"

# Train the model
results = model.train(data="ultralytics/cfg/datasets/Fisheye.yaml", epochs=200, device=device, project=project, batch=16, optimizer='SGD',  name='Fisheye_vml4_')