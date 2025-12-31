import sys
import os
import cv2
import time

# --------------------------------------------------
# Add project root
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolo.inference import YOLOPoseDetector
from tracking.iou_tracker import IOUTracker
from movement.movement_manager import MovementManager
from reporting.report_builder import ReportBuilder

# --------------------------------------------------
# Config
# --------------------------------------------------
VIDEO_PATH = r"D:\Meditation proctor\data\ef81e229\video.mp4"
MAX_SECONDS = 120
DISPLAY_SCALE = 0.7
WAIT_MS = 40
FPS = 25

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
movement_manager = MovementManager(fps=FPS)
report_builder = ReportBuilder()

cap = cv2.VideoCapture(VIDEO_PATH)
start_time = time.time()

print("▶ Movement + Reporting integration test started")

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

        # Face bbox for MediaPipe neck
        face_y2 = y1 + int(0.4 * (y2 - y1))
        face_bbox = (x1, y1, x2, face_y2)

        movement_manager.update(
            person_id=person_id,
            frame=frame,
            keypoints=keypoints,
            face_bbox=face_bbox,
            timestamp=time.time(),
            draw_debug=False
        )

        counts = movement_manager.get_counts(person_id)

        # -------------------------------
        # Draw overlay
        # -------------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, person_id, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y_txt = y2 + 20
        for part in ["neck", "arm", "leg"]:
            cv2.putText(
                frame,
                f"{part}: {counts[part]}",
                (x1, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_txt += 22

    frame_disp = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("Movement + Reporting Test", frame_disp)

    if cv2.waitKey(WAIT_MS) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()

# --------------------------------------------------
# Build final report
# --------------------------------------------------
movement_counts = movement_manager.get_all_counts()
final_report = report_builder.build(movement_counts)


print("\n▶ FINAL PER-PERSON REPORT")
for pid, report in final_report.items():
    print(pid, report)


print("\n✔ Integration test completed successfully")
