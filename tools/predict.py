from ultralytics import YOLO

# Load a model
model = YOLO("results/ultralytics/yolov11/x/fisheye_vml3_/weights/best.pt")  # load a custom model
project = "results/ultralytics/yolov11/x"
name = 'fisheye_vml3_test'
device = [3]

# Define path to the image file
source = "/mnt/vmlqnap01/datasets/Fisheye/test/images/camera1_A_119.png"

# Run inference on the source
results = model.predict(source=source, project=project, name=name, device=device, save=True)
#json_file = results[0].to_json()
# results.save(filename=f"{project}/{name}/results.png")
# # 保存json文件
# with open(f"{project}/{name}/results.json", "w") as f:
#     f.write(json_file)