from ultralytics import YOLO

device = [3]

# Load a model
model = YOLO("yolo11l.yaml")  # build a new model from YAML
model = YOLO("yolo11l.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11l.yaml").load("yolo11l.pt")  # build from YAML and transfer weights

project = "/home/guo/own_ultralytics/results/ultralytics/yolov11/l"

# Train the model
results = model.train(data="/home/guo/own_ultralytics/ultralytics/cfg/datasets/VisDrone.yaml", epochs=300, imgsz=640, device=device, project=project, batch=12, optimizer='SGD',  name='yolo11l_visdrone_vml6_')