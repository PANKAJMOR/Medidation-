import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
from yolo.inference import YOLODetector
from detector_file.person_selector import PersonSelector

# Paths
frame_path = r"D:\Meditation proctor\data\31722b92\frames\00001.jpg"

# Load frame
frame = cv2.imread(frame_path)
if frame is None:
    raise ValueError("❌ Failed to load frame. Check path: " + frame_path)

# Init detector and person selector
detector = YOLODetector("yolov8n.pt")
selector = PersonSelector()

# Run YOLO
detections = detector.detect(frame)
print(f"Detections found: {len(detections)}")

# Select people
result, error = selector.select_people(detections)

if error:
    print("❌ Error:", error)
else:
    print("🎯 Selected people:", result.keys())
