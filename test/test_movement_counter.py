import sys
import os
import cv2
import time

# --------------------------------------------------
# Add project root
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolo.inference import YOLOPoseDetector
from movement.movement_counter import MovementCounter
from tracking.iou_tracker import IOUTracker

# --------------------------------------------------
# Config
# --------------------------------------------------
VIDEO_PATH = r"D:\Meditation proctor\data\ef81e229\video.mp4"
DISPLAY_SCALE = 0.8
MAX_SECONDS = 120
WAIT_MS = 40

# --------------------------------------------------
# YOLOv8 Pose Skeleton
# --------------------------------------------------
SKELETON = [
    (0, 5), (0, 6),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# --------------------------------------------------
# Initialize components
# --------------------------------------------------
detector = YOLOPoseDetector(weights="yolov8n-pose.pt")
tracker = IOUTracker(iou_thresh=0.3)

movement_counter = MovementCounter(
    wrist_thresh=4,
    elbow_thresh=6,
    neck_thresh=5,
    min_active_frames=5,
    min_still_frames=6,
    leg_baseline_frames=25,
    leg_deviation_thresh=30
)

cap = cv2.VideoCapture(VIDEO_PATH)
start_time = time.time()

prev_kpts = {}

print("▶ Movement test started (press 'q' to quit)")

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
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        area = (x2 - x1) * (y2 - y1)

        # Filter small / far persons
        if area < 6000:
            continue

        bbox = [x1, y1, x2, y2]
        bboxes.append(bbox)
        pose_map[tuple(bbox)] = det.keypoints

    # Keep max 3 largest persons
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

        if person_id not in prev_kpts:
            prev_kpts[person_id] = None

        # 🔑 Update movement logic
        movement_counter.update(
            person_id,
            prev_kpts[person_id],
            keypoints
        )
        prev_kpts[person_id] = keypoints

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
        # Draw keypoints
        # -------------------------------
        for i, (x, y) in enumerate(keypoints):
            if x <= 0 or y <= 0:
                continue

            if i == 0:
                color = (0, 0, 255)          # nose
            elif i in [5, 6]:
                color = (0, 255, 255)        # shoulders
            elif i in [7, 8, 9, 10]:
                color = (255, 0, 0)          # arms
            elif i in [11, 12, 13, 14, 15, 16]:
                color = (0, 255, 0)          # legs
            else:
                color = (180, 180, 180)

            cv2.circle(frame, (int(x), int(y)), 4, color, -1)

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
        # Draw movement counters
        # -------------------------------
        counts = movement_counter.get_counts(person_id)
        y_txt = y2 + 20

        for cat in ["arm_hand", "leg"]:
            cv2.putText(
                frame,
                f"{cat}: {counts.get(cat, 0)}",
                (x1, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_txt += 22

    # -------------------------------
    # Display
    # -------------------------------
    frame_disp = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("Meditation Pose + Movement Test (Final)", frame_disp)

    if cv2.waitKey(WAIT_MS) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("\n▶ Final movement counts:")
for pid, counts in movement_counter.movement_count.items():
    print(pid, dict(counts))

print("✔ Test completed successfully")
