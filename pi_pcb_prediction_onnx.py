from ultralytics import YOLO

import torch
torch.cuda.is_available()

model_path = r"/home/oatgz/pcb/pcb.onnx"

# Load a model
model = YOLO(model_path)  # pretrained YOLO11n model

image_test = r"/home/oatgz/pcb/test/images"

# Run batched inference on a list of images
results = model.predict(image_test, imgsz=640)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
