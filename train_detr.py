from ultralytics import RTDETR

device = 3

# Load a model
model = RTDETR('rtdetr-l.yaml')  # build a new model from YAML
# model = RTDETR("rtdetr-l.pt")  # load a pretrained model (recommended for training)
# model = RTDETR("rtdetr-l.yaml").load("rtdetr-l.pt")  # build from YAML and transfer weights

project = "/home/guo/ultralytics/results/ultralytics/RTDETR"

# Train the model
results = model.train(data="/home/guo/ultralytics/ultralytics/cfg/datasets/_visdrone.yaml", epochs=300, imgsz=640, device=device, project=project, batch=4, name = "rtdetr-l_visdrone_vml6")