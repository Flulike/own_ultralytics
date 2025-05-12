from ultralytics import YOLO

device = [1]

# Load a model
model = YOLO("yolo11x.yaml")  # build a new model from YAML
# model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11x.yaml").load("yolo11x.pt")  # build from YAML and transfer weights

project = "results/ultralytics/yolov11/x"

# Train the model
results = model.train(data="ultralytics/cfg/datasets/Fisheye.yaml", epochs=300, imgsz=640, device=device, project=project, batch=12, optimizer='SGD',  name='fisheye_vml6_', pretrained=False)