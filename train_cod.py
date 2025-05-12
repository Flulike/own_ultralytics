from ultralytics import YOLO

device = [2]

# Load a model
model = YOLO("yolo11x.yaml")  # build a new model from YAML
model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x.yaml").load("yolo11x.pt")  # build from YAML and transfer weights

project = "/home/guo/own_ultralytics/results/ultralytics/yolov11/x"

# Train the model
results = model.train(
    data="/home/guo/own_ultralytics/ultralytics/cfg/datasets/CODrone.yaml", 
    epochs=300, 
    device=device, 
    project=project, 
    batch=12, 
    optimizer='SGD',  
    name='codrone_vml6_',)