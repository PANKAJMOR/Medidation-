import sys
import os
import time
import cv2

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from audio.audio_marker import AudioMarker
from yolo.inference import Detection
from movement.movement_counter import MovementCounter

# -------------------------------
# Paths & settings
# -------------------------------
video_path = r"D:\Meditation proctor\data\31722b92\video.mp4"
frames_dir = r"D:\Meditation proctor\data\31722b92\frames"
start_ref = r"D:\Meditation proctor\reference_audio\start_audio.wav"
end_ref = r"D:\Meditation proctor\reference_audio\end_audio.wav"

fps = 1  # frames per second
print("=== Meditation Proctor Pipeline Started ===")

# -------------------------------
# Audio marker: find analysis window
# -------------------------------
print("[1/4] Detecting start/end timestamps based on reference audio...")
am = AudioMarker(min_duration_sec=10600)  # or dynamic as needed

try:
    start_sec, end_sec = am.get_analysis_window(video_path, start_ref, end_ref)
    print(f"[✔] Analysis window: Start at {start_sec:.2f}s, End at {end_sec:.2f}s")
except Exception as e:
    print(f"[❌] Audio analysis failed: {e}")
    sys.exit(1)

# -------------------------------
# Frame extraction
# -------------------------------
print("[2/4] Extracting frames...")
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

cap = cv2.VideoCapture(video_path)
total_frames = 0
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_sec = frame_count / fps
    if curr_sec < start_sec:
        frame_count += 1
        continue
    if curr_sec > end_sec:
        break

    frame_file = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(frame_file, frame)
    frame_count += 1
    total_frames += 1
    if frame_count % 50 == 0:
        print(f"[ℹ] Extracted {frame_count} frames...")

cap.release()
print(f"[✔] Total frames extracted: {total_frames}")

# -------------------------------
# Initialize YOLO & movement counter
# -------------------------------
print("[3/4] Initializing detection and movement modules...")
detector = Detection()
mc = MovementCounter()

# Define person regions (example; should be dynamic later)
person_boxes = {
    "left": [100, 200, 300, 600],
    "center": [350, 200, 550, 600],
    "right": [600, 200, 800, 600]
}

# -------------------------------
# Movement analysis
# -------------------------------
print("[4/4] Starting movement analysis...")
movements = {pos: {"leg":0, "arm_hand":0, "neck":0} for pos in person_boxes.keys()}
prev_kpts = {pos: None for pos in person_boxes.keys()}

frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
for i, frame_file in enumerate(frame_files):
    frame = cv2.imread(os.path.join(frames_dir, frame_file))
    results = detector.detect(frame)

    for pos, box in person_boxes.items():
        keypoints = detector.extract_keypoints(results, box)

        # Count movements
        if prev_kpts[pos] is not None:
            for cat in ["leg", "arm_hand", "neck"]:
                movements[pos][cat] += mc.calculate_movement(prev_kpts[pos], keypoints, cat)
        prev_kpts[pos] = keypoints

    if i % 50 == 0:
        print(f"[ℹ] Processed {i}/{len(frame_files)} frames...")

print("[✔] Movement analysis completed!")
print("Final movements:", movements)
print("=== Pipeline Finished ===")
