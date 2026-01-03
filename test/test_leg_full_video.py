import sys
import os
import cv2
import time
import json
from collections import defaultdict

# --------------------------------------------------
# Add project root
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolo.inference import YOLOPoseDetector
from movement.leg import LegMovement
from tracking.iou_tracker import IOUTracker

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
VIDEO_PATH = r"D:\Pankaj\Meditation proctor Main\videos\eef35db0\video.mp4"
DISPLAY_SCALE = 0.8
FPS_PROCESS = 1          # 1 FPS for long video
PRINT_INTERVAL_SEC = 30  # Console logging interval

# --------------------------------------------------
# INITIALIZE
# --------------------------------------------------
detector = YOLOPoseDetector(
    weights="yolov8n-pose.pt",
    conf=0.4,
    imgsz=640
)

tracker = IOUTracker(iou_thresh=0.3)

leg_movement = LegMovement(
    # ankle_thresh=3,
    # knee_dist_thresh=5,
    # hold_seconds=0.5,
    # fps=FPS_PROCESS,
    # stable_frames=25
    ankle_thresh=6,
    knee_dist_thresh=10,
    hold_seconds=5.0,
    fps=1,
    stable_frames=10
)

cap = cv2.VideoCapture(VIDEO_PATH)
video_fps = cap.get(cv2.CAP_PROP_FPS)
FRAME_STRIDE = int(video_fps / FPS_PROCESS)
frame_idx = 0

# Count storage for display
leg_counts = defaultdict(int)
last_signal = defaultdict(lambda: "-")

last_print_sec = 0

print("▶ LEG MOVEMENT VISUAL + JSON TEST STARTED")
print(f"▶ Processing at {FPS_PROCESS} FPS\n")

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    if frame_idx % FRAME_STRIDE != 0:
        continue

    timestamp_sec = frame_idx / video_fps
    timestamp_txt = time.strftime("%H:%M:%S", time.gmtime(timestamp_sec))

    detections = detector.detect(frame)

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

    bboxes = sorted(
        bboxes,
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        reverse=True
    )[:3]

    tracked = tracker.update(bboxes)

    # -------------------------
    # PROCESS PERSONS
    # -------------------------
    for track_id, bbox in tracked:
        person_id = f"person_{track_id}"
        keypoints = pose_map.get(tuple(bbox))
        if keypoints is None:
            continue

        signal = leg_movement.update(person_id, keypoints)

        if signal == "START":
            leg_counts[person_id] += 1
            last_signal[person_id] = "START"

        elif signal == "END":
            last_signal[person_id] = "END"

        x1, y1, x2, y2 = bbox

        # -------------------------
        # DRAW
        # -------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Ankles
        for idx in [15, 16]:
            xk, yk = keypoints[idx]
            if xk > 0 and yk > 0:
                cv2.circle(frame, (int(xk), int(yk)), 6, (0, 0, 255), -1)

        # Knees
        for idx in [13, 14]:
            xk, yk = keypoints[idx]
            if xk > 0 and yk > 0:
                cv2.circle(frame, (int(xk), int(yk)), 5, (255, 0, 0), -1)

        state = leg_movement.state[person_id]

        cv2.putText(
            frame,
            f"{person_id}",
            (x1, y1 - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"LEG COUNT: {leg_counts[person_id]}",
            (x1, y1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"STATE: {state} | {last_signal[person_id]}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    # Timestamp overlay
    cv2.putText(
        frame,
        f"TIME: {timestamp_txt}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # -------------------------
    # CONSOLE JSON LOGGING
    # -------------------------
    if int(timestamp_sec) - last_print_sec >= PRINT_INTERVAL_SEC:
        last_print_sec = int(timestamp_sec)

        print(json.dumps({
            "video_time": timestamp_txt,
            "leg_counts": dict(leg_counts),
            "states": dict(leg_movement.state),
            "last_signal": dict(last_signal)
        }, indent=2))

    frame_disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("LEG MOVEMENT – VISUAL + COUNT", frame_disp)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("\n▶ FINAL LEG MOVEMENT COUNTS")
print(json.dumps(dict(leg_counts), indent=2))
print("✔ TEST COMPLETED")
