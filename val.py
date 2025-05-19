from ultralytics import YOLO

# Load a model
model = YOLO("results/ultralytics/yolov11/x/fisheye_vml3_/weights/best.pt")  # load a custom model
project = "results/ultralytics/yolov11/x"
name = 'fisheye_vml3_test'
device = [3]
data="ultralytics/cfg/datasets/Fisheye.yaml"
# Validate the model
metrics = model.val(data=data, save_json=True, project=project, name=name, iou=0.5, conf=0.5, device=device) 