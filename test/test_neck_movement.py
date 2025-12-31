import sys
import os
import cv2
import time

# --------------------------------------------------
# Add project root
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolo.inference import YOLOPoseDetector
from movement.neck import NeckMovement
from tracking.iou_tracker import IOUTracker

# --------------------------------------------------
# Config
# --------------------------------------------------
VIDEO_PATH = r"D:\Meditation proctor\data\ef81e229\video.mp4"
DISPLAY_SCALE = 0.8
MAX_SECONDS = 120
WAIT_MS = 40

# --------------------------------------------------
# Upper-body skeleton (for visualization)
# --------------------------------------------------
SKELETON = [
    (0, 5), (0, 6),
    (5, 6)
]

# --------------------------------------------------
# Initialize components
# --------------------------------------------------
detector = YOLOPoseDetector(
    weights="yolov8n-pose.pt",
    imgsz=640
)

tracker = IOUTracker(iou_thresh=0.3)

# 🔑 UPDATED neck movement (yaw + nose Y)
neck_movement = NeckMovement(
    yaw_thresh= 1,        # horizontal movement
    nose_y_thresh=1,     # vertical movement
    hold_seconds=0.5,
    fps=25,
    min_still_frames=6
)

cap = cv2.VideoCapture(VIDEO_PATH)
start_time = time.time()

print("▶ Neck movement test started (press 'q' to quit)")

# --------------------------------------------------
# Main loop
# --------------------------------------------------
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    if time.time() - start_time > MAX_SECONDS:
        break

    detections = detector.detect(frame)

    # -------------------------------
    # Prepare bounding boxes
    # -------------------------------
    bboxes = []
    pose_map = {}

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        area = (x2 - x1) * (y2 - y1)

        # Filter small / far persons
        if area < 6000:
            continue

        bbox = [int(x1), int(y1), int(x2), int(y2)]
        bboxes.append(bbox)
        pose_map[tuple(bbox)] = det.keypoints

    # Keep max 3 closest persons
    bboxes = sorted(
        bboxes,
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        reverse=True
    )[:3]

    tracked = tracker.update(bboxes)

    # -------------------------------
    # Process tracked persons
    # -------------------------------
    for track_id, bbox in tracked:
        x1, y1, x2, y2 = bbox
        person_id = f"person_{track_id}"

        keypoints = pose_map.get(tuple(bbox))
        if keypoints is None:
            continue

        # 🔑 Update NECK movement
        neck_movement.update(person_id, keypoints)

        # -------------------------------
        # Draw bounding box + ID
        # -------------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            person_id,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # -------------------------------
        # Draw keypoints (nose + shoulders)
        # -------------------------------
        for idx, color in [
            (0, (0, 0, 255)),   # nose (RED)
            (5, (0, 255, 255)), # left shoulder
            (6, (0, 255, 255))  # right shoulder
        ]:
            xk, yk = keypoints[idx]
            if xk > 0 and yk > 0:
                cv2.circle(frame, (int(xk), int(yk)), 6, color, -1)

        # -------------------------------
        # Draw skeleton
        # -------------------------------
        for p1, p2 in SKELETON:
            x1k, y1k = keypoints[p1]
            x2k, y2k = keypoints[p2]
            if x1k > 0 and y1k > 0 and x2k > 0 and y2k > 0:
                cv2.line(
                    frame,
                    (int(x1k), int(y1k)),
                    (int(x2k), int(y2k)),
                    (255, 255, 255),
                    2
                )

        # -------------------------------
        # Draw NECK movement count
        # -------------------------------
        count = neck_movement.get_count(person_id)
        cv2.putText(
            frame,
            f"neck: {count}",
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # -------------------------------
    # Display
    # -------------------------------
    frame_disp = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("Neck Movement Test (Yaw + Vertical)", frame_disp)

    if cv2.waitKey(WAIT_MS) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("\n▶ Final neck movement counts:")
for pid, cnt in neck_movement.count.items():
    print(pid, cnt)

print("✔ Neck test completed successfully")
