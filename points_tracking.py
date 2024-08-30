from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")

# For pointing Video
result = model(
    source="/home/hany_jr/Downloads/WhatSie/WhatsApp Video 2024-08-30 at 3.39.35 AM.mp4",
    show=True,
    conf=0.3,
    save=True,
)

# For pointing images
result = model(
    source="/home/hany_jr/Downloads/images.jpeg",
    show=True,
    conf=0.3,
    save=True,
)
