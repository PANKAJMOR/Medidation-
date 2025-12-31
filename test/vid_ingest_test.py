import sys
import os

# add parent dir to path so "ingestion" becomes importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.video_ingestion import VideoIngestion

vi = VideoIngestion()

src = "https://www.youtube.com/live/Pel6lO8qibg?si=EEIpviTg__2xglev"  # replace with working link

video_input, video_id, mode = vi.ingest(src)

print("Video ID:", video_id)
print("Mode used:", mode)

frames = vi.extract_frames(video_input, video_id, mode, fps=1)

print("Frames saved:", frames)
