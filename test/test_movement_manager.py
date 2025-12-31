# import sys
# import os
# import cv2
# import time

# # --------------------------------------------------
# # Add project root
# # --------------------------------------------------
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from yolo.inference import YOLOPoseDetector
# from tracking.iou_tracker import IOUTracker
# from movement.movement_manager import MovementManager

# # --------------------------------------------------
# # Config
# # --------------------------------------------------
# VIDEO_PATH = r"D:\Meditation proctor\data\9f0de88e\video.mp4.mkv"
# DISPLAY_SCALE = 0.8
# MAX_SECONDS = 125
# WAIT_MS = 40
# FPS = 25

# # --------------------------------------------------
# # Initialize components
# # --------------------------------------------------
# detector = YOLOPoseDetector(
#     weights="yolov8n-pose.pt",
#     conf=0.4,
#     iou=0.5,
#     imgsz=640
# )

# tracker = IOUTracker(iou_thresh=0.3)
# movement_manager = MovementManager(fps=FPS)

# cap = cv2.VideoCapture(VIDEO_PATH)
# start_time = time.time()

# print("▶ Full movement test started (press 'q' to quit)")

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

#         # Face crop: upper 40% of bbox
#         face_y2 = y1 + int(0.4 * (y2 - y1))
#         face_bbox = (x1, y1, x2, face_y2)

#         # 🔑 Update all movements
#         movement_manager.update(
#             person_id=person_id,
#             frame=frame,
#             keypoints=keypoints,
#             face_bbox=face_bbox,
#             timestamp=time.time(),
#             draw_debug=False   # set True to see neck landmarks
#         )

#         counts = movement_manager.get_counts(person_id)

#         # -------------------------------
#         # Draw person bbox + ID
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
#         # Draw movement counters
#         # -------------------------------
#         y_txt = y2 + 22
#         for cat in ["neck", "arm", "leg"]:
#             cv2.putText(
#                 frame,
#                 f"{cat}: {counts[cat]}",
#                 (x1, y_txt),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65,
#                 (0, 255, 0),
#                 2
#             )
#             y_txt += 22

#     # -------------------------------
#     # Display
#     # -------------------------------
#     frame_disp = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
#     cv2.imshow("Meditation – Full Movement Test", frame_disp)

#     if cv2.waitKey(WAIT_MS) & 0xFF == ord("q"):
#         break

# # --------------------------------------------------
# # Cleanup
# # --------------------------------------------------
# cap.release()
# cv2.destroyAllWindows()

# print("\n▶ Final movement counts:")
# for pid, counts in movement_manager.get_all_counts().items():
#     print(pid, counts)

# print("✔ Full movement test completed successfully")

import sys
import os
import cv2
import time
import json

# --------------------------------------------------
# Add project root
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolo.inference import YOLOPoseDetector
from tracking.iou_tracker import IOUTracker
from movement.movement_manager import MovementManager

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
VIDEO_PATH = r"D:\Meditation proctor\data\ef81e229\video.mp4"
DISPLAY_SCALE = 0.8
MAX_SECONDS = 120
FPS = 25
FRAME_STRIDE = FPS   # 1 FPS processing (stable + fast)

# --------------------------------------------------
# INIT
# --------------------------------------------------
detector = YOLOPoseDetector(
    weights="yolov8n-pose.pt",
    conf=0.4,
    imgsz=640
)

tracker = IOUTracker(iou_thresh=0.3)
movement_manager = MovementManager(fps=FPS)

cap = cv2.VideoCapture(VIDEO_PATH)
start_time = time.time()
frame_idx = 0

print("\n▶ MovementManager test started (press 'q' to quit)\n")

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # stop after MAX_SECONDS
    if time.time() - start_time > MAX_SECONDS:
        break

    # stride
    if frame_idx % FRAME_STRIDE != 0:
        continue

    timestamp = time.time()

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

    # keep top 3 persons
    bboxes = sorted(
        bboxes,
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        reverse=True
    )[:3]

    tracked = tracker.update(bboxes)

    # -------------------------------
    # PROCESS PERSONS
    # -------------------------------
    for track_id, bbox in tracked:
        x1, y1, x2, y2 = bbox
        person_id = f"person_{track_id}"

        keypoints = pose_map.get(tuple(bbox))
        if keypoints is None:
            continue

        # Face bbox = upper 40%
        face_y2 = y1 + int(0.4 * (y2 - y1))
        face_bbox = (x1, y1, x2, face_y2)

        # 🔑 UPDATE MOVEMENT MANAGER
        movement_manager.update(
            person_id=person_id,
            frame=frame,
            keypoints=keypoints,
            face_bbox=face_bbox,
            frame_sec=timestamp
        )

        counts = movement_manager.get_counts(person_id)

        # -------------------------------
        # DRAW
        # -------------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            person_id,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        y_txt = y2 + 20
        for k in ["neck", "arm", "leg"]:
            cv2.putText(
                frame,
                f"{k}: {counts[k]}",
                (x1, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_txt += 20

    # -------------------------------
    # DISPLAY
    # -------------------------------
    frame_disp = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("MovementManager Test", frame_disp)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()

# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------
print("\n▶ FINAL COUNTS")
print(json.dumps(movement_manager.get_all_counts(), indent=2))

print("\n▶ TIMESTAMPS")
print(json.dumps(movement_manager.get_timestamps(), indent=2))

print("\n✔ MovementManager test completed successfully")
