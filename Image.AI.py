import cv2
import torch
import numpy as np
# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can use other model variants like 'yolov5m' or 'yolov5l'
# Set webcam source (usually 0 for the default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Error: Could not open the webcam.")
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if there's an issue reading the frame
    results = model(frame)
    image = results.render()[0]
    cv2.imshow('image', image)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()