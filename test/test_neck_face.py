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
# from movement.neck_face import FaceNeckMovement

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
#     conf=0.4,
#     iou=0.5,
#     imgsz=640
# )

# tracker = IOUTracker(iou_thresh=0.3)

# neck_face = FaceNeckMovement(
#     yaw_delta_thresh=8.0,      # degrees CHANGE
#     pitch_delta_thresh=8.0,
#     hold_frames=3,
#     cooldown_seconds=0.6,
#     fps=25
# )

# cap = cv2.VideoCapture(VIDEO_PATH)
# start_time = time.time()

# print("▶ MediaPipe neck test started (press 'q' to quit)")

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
#         pose_map[tuple(bbox)] = det

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

#         det = pose_map.get(tuple(bbox))
#         if det is None:
#             continue

#         # Face crop: upper 40% of person bbox
#         face_y2 = y1 + int(0.4 * (y2 - y1))
#         face_bbox = (x1, y1, x2, face_y2)

#         # 🔑 Update MediaPipe neck
#         neck_face.update(person_id, frame, face_bbox, draw = True)

#         # Draw person bbox
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

#         # Draw face bbox
#         fx1, fy1, fx2, fy2 = face_bbox
#         cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)

#         # Draw neck count
#         count = neck_face.get_count(person_id)
#         cv2.putText(
#             frame,
#             f"neck: {count}",
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
#     cv2.imshow("MediaPipe Neck Test (Multi-Person)", frame_disp)

#     if cv2.waitKey(WAIT_MS) & 0xFF == ord("q"):
#         break

# # --------------------------------------------------
# # Cleanup
# # --------------------------------------------------
# cap.release()
# cv2.destroyAllWindows()

# print("\n▶ Final neck movement counts:")
# for pid, cnt in neck_face.count.items():
#     print(pid, cnt)

# print("✔ MediaPipe neck test completed successfully")



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
from tracking.iou_tracker import IOUTracker
from movement.neck_face import FaceNeckMovement

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
    conf=0.4,
    iou=0.5,
    imgsz=640
)

tracker = IOUTracker(iou_thresh=0.3)

neck_face = FaceNeckMovement(
    yaw_delta_thresh=8.0,
    pitch_delta_thresh=8.0,
    hold_frames=3,
    cooldown_seconds=0.6,
    fps=25
)

# --------------------------------------------------
# State (TEST SIDE)
# --------------------------------------------------
neck_counts = defaultdict(int)
neck_state = defaultdict(lambda: "STILL")

cap = cv2.VideoCapture(VIDEO_PATH)
start_time = time.time()

print("▶ MediaPipe neck test started (press 'q' to quit)")

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
        pose_map[tuple(bbox)] = det

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

        det = pose_map.get(tuple(bbox))
        if det is None:
            continue

        # Face crop (upper 40%)
        face_y2 = y1 + int(0.4 * (y2 - y1))
        face_bbox = (x1, y1, x2, face_y2)

        # 🔑 UPDATE NECK
        signal = neck_face.update(
            person_id,
            frame,
            face_bbox,
            draw=True
        )

        if signal == "START":
            neck_counts[person_id] += 1
            neck_state[person_id] = "MOVING"
            print(f"[{person_id}] NECK START → count={neck_counts[person_id]}")

        elif signal == "END":
            neck_state[person_id] = "STILL"
            print(f"[{person_id}] NECK END")

        # -------------------------------
        # Draw person bbox + label
        # -------------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{person_id} | neck={neck_counts[person_id]} | {neck_state[person_id]}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # Draw face bbox
        fx1, fy1, fx2, fy2 = face_bbox
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)

    # -------------------------------
    # Display
    # -------------------------------
    frame_disp = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("MediaPipe Neck Test (START / END)", frame_disp)

    if cv2.waitKey(WAIT_MS) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("\n▶ Final neck movement counts:")
for pid, cnt in neck_counts.items():
    print(pid, cnt)

print("✔ MediaPipe neck test completed successfully")
