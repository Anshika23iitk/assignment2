import numpy as np
import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)


# Path to input video
video_path = "C:\\Users\\ADMIN\\Downloads\\person-bicycle-car-detection.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties for output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("C:\\Users\\ADMIN\\Downloads\\output_video_yolo.mp4", fourcc, fps, (width, height))
# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection using YOLOv5
    results = model(frame)

    # Get detection results
    detections = results.xyxy[0]  # Bounding boxes, confidence, and class info

    # Draw bounding boxes and labels on the frame
    for *box, conf, cls in detections:
        if conf >= CONFIDENCE_THRESHOLD:  # Filter based on confidence
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Write the annotated frame to the output video
    out.write(frame)

    # Display the frame in real-time (optional)
    cv2.imshow("YOLOv5 Video Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()