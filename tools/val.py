from ultralytics import YOLO

# Load a model
model = YOLO("results/ultralytics/yolov11/x/yolo11x_visdrone_vml6_/weights/best.pt")  # load a custom model
project = "results/visapp"
name = 'visdrone_ours'
device = [2]
data="ultralytics/cfg/datasets/VisDrone.yaml"
# Validate the model
metrics = model.val(data=data, save_json=True, project=project, name=name, device=device, visualize=True) 