# import sys
# import os
# import cv2
# import time

# # --------------------------------------------------
# # Add project root
# # --------------------------------------------------
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from yolo.inference import YOLOPoseDetector
# from movement.arm import ArmMovement
# from tracking.iou_tracker import IOUTracker

# # --------------------------------------------------
# # Config
# # --------------------------------------------------
# VIDEO_PATH = r"D:\Meditation proctor\data\ef81e229\video.mp4"
# DISPLAY_SCALE = 0.8
# MAX_SECONDS = 120
# WAIT_MS = 40

# # --------------------------------------------------
# # Initialize components
# # --------------------------------------------------
# detector = YOLOPoseDetector(
#     weights="yolov8n-pose.pt",
#     imgsz=640
# )

# tracker = IOUTracker(iou_thresh=0.3)

# arm_movement = ArmMovement(
#     wrist_thresh=4,
#     elbow_thresh=6,
#     hold_seconds=0.6,
#     fps=25,
#     lap_margin=20
# )

# cap = cv2.VideoCapture(VIDEO_PATH)
# start_time = time.time()

# print("▶ Arm movement test started (press 'q' to quit)")

# # --------------------------------------------------
# # Main loop
# # --------------------------------------------------
# while cap.isOpened():

#     ret, frame = cap.read()
#     if not ret:
#         break

#     if time.time() - start_time > MAX_SECONDS:
#         break

#     detections = detector.detect(frame)

#     # -------------------------------
#     # Prepare bounding boxes
#     # -------------------------------
#     bboxes = []
#     pose_map = {}

#     for det in detections:
#         x1, y1, x2, y2 = det.bbox
#         area = (x2 - x1) * (y2 - y1)

#         # Filter small / far persons
#         if area < 6000:
#             continue

#         bbox = [x1, y1, x2, y2]
#         bboxes.append(bbox)
#         pose_map[tuple(bbox)] = det.keypoints

#     # Keep max 3 closest persons
#     bboxes = sorted(
#         bboxes,
#         key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
#         reverse=True
#     )[:3]

#     tracked = tracker.update(bboxes)

#     # -------------------------------
#     # Process tracked persons
#     # -------------------------------
#     for track_id, bbox in tracked:
#         x1, y1, x2, y2 = bbox
#         person_id = f"person_{track_id}"

#         keypoints = pose_map.get(tuple(bbox))
#         if keypoints is None:
#             continue

#         # 🔑 Update ARM movement
#         arm_movement.update(person_id, keypoints)

#         # -------------------------------
#         # Draw bounding box + ID
#         # -------------------------------
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(
#             frame,
#             person_id,
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 255),
#             2
#         )

#         # -------------------------------
#         # Draw keypoints (arms + hips)
#         # -------------------------------
#         # wrists, elbows, hips
#         for idx, color in [
#             (7, (255, 255, 0)),   # left elbow
#             (8, (255, 255, 0)),   # right elbow
#             (9, (0, 0, 255)),     # left wrist
#             (10, (0, 0, 255)),    # right wrist
#             (11, (0, 255, 0)),    # left hip
#             (12, (0, 255, 0))     # right hip
#         ]:
#             x, y = keypoints[idx]
#             if x > 0 and y > 0:
#                 cv2.circle(frame, (int(x), int(y)), 5, color, -1)

#         # -------------------------------
#         # Draw ARM movement counter
#         # -------------------------------
#         count = arm_movement.get_count(person_id)
#         cv2.putText(
#             frame,
#             f"arm: {count}",
#             (x1, y2 + 25),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 0),
#             2
#         )

#     # -------------------------------
#     # Display
#     # -------------------------------
#     frame_disp = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
#     cv2.imshow("Arm Movement Test", frame_disp)

#     if cv2.waitKey(WAIT_MS) & 0xFF == ord("q"):
#         break

# # --------------------------------------------------
# # Cleanup
# # --------------------------------------------------
# cap.release()
# cv2.destroyAllWindows()

# print("\n▶ Final arm movement counts:")
# for pid, cnt in arm_movement.count.items():
#     print(pid, cnt)

# print("✔ Arm test completed successfully")


import sys
import os
import cv2
import time
from collections import defaultdict

# --------------------------------------------------
# Add project root
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolo.inference import YOLOPoseDetector
from movement.arm import ArmMovement
from tracking.iou_tracker import IOUTracker

# --------------------------------------------------
# Config
# --------------------------------------------------
VIDEO_PATH = r"D:\Meditation proctor\data\ef81e229\video.mp4"
DISPLAY_SCALE = 0.8
MAX_SECONDS = 120
WAIT_MS = 40

# --------------------------------------------------
# Initialize components
# --------------------------------------------------
detector = YOLOPoseDetector(
    weights="yolov8n-pose.pt",
    imgsz=640
)

tracker = IOUTracker(iou_thresh=0.3)

arm_movement = ArmMovement(
    wrist_thresh=4,
    elbow_thresh=6,
    hold_seconds=0.6,
    fps=25,
    lap_margin=20
)

# -------------------------------
# TEST-SIDE STATE
# -------------------------------
arm_counts = defaultdict(int)
arm_state = defaultdict(lambda: "STILL")

cap = cv2.VideoCapture(VIDEO_PATH)
start_time = time.time()

print("▶ Arm movement test started (press 'q' to quit)")

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

        if area < 6000:
            continue

        bbox = [x1, y1, x2, y2]
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

        # 🔑 UPDATE ARM (START / END)
        signal = arm_movement.update(person_id, keypoints)

        if signal == "START":
            arm_counts[person_id] += 1
            arm_state[person_id] = "MOVING"
            print(f"[{person_id}] ARM START → count={arm_counts[person_id]}")

        elif signal == "END":
            arm_state[person_id] = "STILL"
            print(f"[{person_id}] ARM END")

        # -------------------------------
        # Draw bounding box + ID
        # -------------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{person_id} | arm={arm_counts[person_id]} | {arm_state[person_id]}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # -------------------------------
        # Draw keypoints (arms + hips)
        # -------------------------------
        for idx, color in [
            (7, (255, 255, 0)),   # left elbow
            (8, (255, 255, 0)),   # right elbow
            (9, (0, 0, 255)),     # left wrist
            (10, (0, 0, 255)),    # right wrist
            (11, (0, 255, 0)),    # left hip
            (12, (0, 255, 0))     # right hip
        ]:
            xk, yk = keypoints[idx]
            if xk > 0 and yk > 0:
                cv2.circle(frame, (int(xk), int(yk)), 5, color, -1)

    # -------------------------------
    # Display
    # -------------------------------
    frame_disp = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("Arm Movement Test (START / END)", frame_disp)

    if cv2.waitKey(WAIT_MS) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("\n▶ Final arm movement counts:")
for pid, cnt in arm_counts.items():
    print(pid, cnt)

print("✔ Arm test completed successfully")
