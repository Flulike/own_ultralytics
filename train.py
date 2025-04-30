from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Display model information (optional)
model.info()

device = [2]

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/home/guo/own_ultralytics/ultralytics/cfg/datasets/_visdrone.yaml", epochs=100, imgsz=640,project='results/ultralytics/RTDETR',
                name='rtdetr-l_visdrone_vml6_',)

# # Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")