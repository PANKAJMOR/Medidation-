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
from identity.role_assigner import RoleAssigner
from movement.movement_manager import MovementManager
from audio.audio_marker import AudioMarker

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
VIDEO_PATH = r"D:\Meditation proctor\data\bfcca4dc\video.mp4.mkv"
DISPLAY_SCALE = 0.9
FPS_OVERRIDE = None   # set to 25 if OpenCV fps is wrong
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --------------------------------------------------
# INIT
# --------------------------------------------------
detector = YOLOPoseDetector(
    weights="yolov8n-pose.pt",
    conf=0.6,
    imgsz=640
)

tracker = IOUTracker(iou_thresh=0.3)
role_assigner = RoleAssigner()
movement_manager = MovementManager(fps=25)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = FPS_OVERRIDE or cap.get(cv2.CAP_PROP_FPS)

audio_marker = AudioMarker()
start_sec, end_sec = audio_marker.get_analysis_window(
    VIDEO_PATH,
    start_ref=r"D:\Meditation proctor\reference_audio\start_audio.wav",
    end_ref=r"D:\Meditation proctor\reference_audio\end_audio.wav"
)

start_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)

frame_idx = 0
FRAME_STRIDE = int(fps)  # 1 FPS processing

participant_last_seen = {}

print("\nVISUAL ANALYSIS STARTED")
print(f"Analysis window: {start_sec:.2f}s → {end_sec:.2f}s")
print("Press Q to quit\n")

# --------------------------------------------------
# LOOP
# --------------------------------------------------
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Skip before window
    if frame_idx < start_frame:
        continue

    if frame_idx > end_frame:
        break

    # 1 FPS processing
    if frame_idx % FRAME_STRIDE != 0:
        continue

    timestamp_sec = (frame_idx - start_frame) / fps

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

    # ROLE ASSIGNMENT
    tracked_people = [(f"person_{tid}", bbox) for tid, bbox in tracked]
    if not role_assigner.assigned and tracked_people:
        role_assigner.assign(tracked_people)

    # DRAW ANALYSIS INFO
    cv2.putText(
        frame,
        f"Video Time: {timestamp_sec:.1f}s",
        (10, 30),
        FONT, 0.8, (0, 255, 255), 2
    )

    for track_id, bbox in tracked:
        x1, y1, x2, y2 = bbox
        person_id = f"person_{track_id}"
        role = role_assigner.role_map.get(person_id, "UNASSIGNED")
        counts = movement_manager.get_counts(person_id)

        # FACE BBOX
        face_y2 = y1 + int(0.4 * (y2 - y1))
        face_bbox = (x1, y1, x2, face_y2)

        # UPDATE MOVEMENTS
        movement_manager.update(
            person_id=person_id,
            frame=frame,
            keypoints=pose_map.get(tuple(bbox)),
            face_bbox=face_bbox,
            frame_sec=timestamp_sec
        )

        participant_last_seen[person_id] = frame_idx

        # DRAW PERSON
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{person_id} | {role}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    FONT, 0.6, (255, 255, 0), 2)

        y_txt = y2 + 20
        for k in ["neck", "arm", "leg"]:
            cv2.putText(
                frame,
                f"{k}: {counts[k]}",
                (x1, y_txt),
                FONT, 0.55, (0, 255, 0), 2
            )
            y_txt += 18

    # DISCONTINUITY VISUAL
    for pid, role in role_assigner.role_map.items():
        last_seen = participant_last_seen.get(pid)
        if last_seen and frame_idx - last_seen > int(15 * fps):
            cv2.putText(
                frame,
                f"{pid} ({role}) DISCONTINUED",
                (10, 60),
                FONT, 0.7, (0, 0, 255), 2
            )

    # SHOW
    frame_disp = cv2.resize(
        frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE
    )
    cv2.imshow("Visual Analyze Video Debug", frame_disp)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nVISUAL ANALYSIS FINISHED")
