import numpy as np
import torch
import torchvision
import cv2
from torchvision.models.detection import SSD300_VGG16_Weights

# Load the model with the updated weight enum
model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
model.eval()

# Print total parameters
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Video path
video_path = "C:\\Users\\ADMIN\\Downloads\\person-bicycle-car-detection.mp4"

# Open video file
cap = cv2.VideoCapture(video_path)

# Get video properties for saving the output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("C:\\Users\\ADMIN\\Downloads\\output_video_ssd.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PyTorch tensor
    image_tensor = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0)

    # Perform detection
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract predictions
    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']
    labels = outputs[0]['labels']

    # Draw detections with score > 0.7
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.7:
            x1, y1, x2, y2 = box.int().tolist()
            class_name = f"Class {label}"  # Replace with actual class name if available
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame in real-time (optional)
    cv2.imshow("Video Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
