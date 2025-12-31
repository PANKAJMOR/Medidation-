import os
import cv2
import json
import uuid
import platform
import subprocess
import yt_dlp

class VideoIngestion:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    # -------------------------------
    # Generate unique video ID
    # -------------------------------
    def generate_video_id(self):
        return str(uuid.uuid4())[:8]

    # -------------------------------
    # Check input type
    # -------------------------------
    def is_youtube(self, src):
        return "youtube.com" in src or "youtu.be" in src

    def is_drive(self, src):
        return "drive.google.com" in src

    # -------------------------------
    # Resolve Google Drive direct link
    # -------------------------------
    def resolve_drive_url(self, url):
        try:
            file_id = url.split("/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        except:
            return url

    # -------------------------------
    # Download video using yt-dlp
    # -------------------------------
    def download_video(self, src, output_path):
        opts = {
            "outtmpl": output_path,
            "quiet": False,  # Show download progress
            "format": "best[ext=mp4]/best",  # Prefer mp4
        }
        
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(src, download=True)
                
                # yt-dlp might change the extension, so find the actual file
                actual_filename = ydl.prepare_filename(info)
                
                print(f"yt-dlp saved file as: {actual_filename}")
                
                # If the filename is different from expected, use the actual one
                if os.path.exists(actual_filename):
                    return os.path.abspath(os.path.normpath(actual_filename))
                
                # Otherwise check if output_path exists
                if os.path.exists(output_path):
                    return os.path.abspath(os.path.normpath(output_path))
                
                # Check for common extensions yt-dlp might add
                for ext in ['.mkv', '.webm', '.m4a', '.mp4']:
                    check_path = output_path + ext
                    if os.path.exists(check_path):
                        print(f"Found video with extension: {ext}")
                        return os.path.abspath(os.path.normpath(check_path))
                
                # Last resort: check the directory for any video file
                video_dir = os.path.dirname(output_path)
                files = os.listdir(video_dir)
                print(f"Files in directory: {files}")
                
                for f in files:
                    if f.startswith('video') and any(f.endswith(e) for e in ['.mp4', '.mkv', '.webm']):
                        found_path = os.path.join(video_dir, f)
                        print(f"Found video file: {found_path}")
                        return os.path.abspath(os.path.normpath(found_path))
                
                return None
                
        except Exception as e:
            print(f"Download error: {e}")
            return None

    # -------------------------------
    # Try streaming (Linux/Cloud only)
    # -------------------------------
    def try_stream(self, src):
        try:
            # Only allow streaming on non-Windows
            if platform.system() != "Windows":
                command = [
                    "yt-dlp",
                    "-f", "best",
                    "-o", "-",
                    src
                ]
                pipe = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                return pipe
        except Exception:
            return None

    # -------------------------------
    # Main ingestion logic
    # -------------------------------
    def ingest(self, src):
        video_id = self.generate_video_id()
        video_dir = os.path.join(self.base_dir, video_id)
        os.makedirs(video_dir, exist_ok=True)

        local_video_path = os.path.join(video_dir, "video.mp4")
        # Normalize the path immediately
        local_video_path = os.path.abspath(os.path.normpath(local_video_path))

        # Case 1: Local file
        if os.path.exists(src):
            mode = "local"
            # Return absolute normalized path
            return os.path.abspath(os.path.normpath(src)), video_id, mode

        # Case 2: YouTube
        if self.is_youtube(src):
            # Try streaming on non-Windows
            stream = self.try_stream(src)
            if stream:
                return stream, video_id, "stream"

            # Otherwise download
            downloaded_path = self.download_video(src, local_video_path)
            # Verify download succeeded
            if not downloaded_path or not os.path.exists(downloaded_path):
                print(f"❌ Download failed: file does not exist")
                return None, None, None
            print(f"✅ Downloaded successfully to: {downloaded_path}")
            return downloaded_path, video_id, "download"

        # Case 3: Google Drive
        if self.is_drive(src):
            clean_url = self.resolve_drive_url(src)
            downloaded_path = self.download_video(clean_url, local_video_path)
            # Verify download succeeded
            if not downloaded_path or not os.path.exists(downloaded_path):
                print(f"❌ Download failed: file does not exist")
                return None, None, None
            print(f"✅ Downloaded successfully to: {downloaded_path}")
            return downloaded_path, video_id, "download"

        return None, None, None

    # -------------------------------
    # Extract frames at given FPS
    # -------------------------------
    def extract_frames(self, video_input, video_id, mode, fps=1):
        frame_dir = os.path.join(self.base_dir, video_id, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        saved = 0

        if mode in ["download", "local"]:
            cap = cv2.VideoCapture(video_input)
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            interval = max(int(video_fps / fps), 1)

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % interval == 0:
                    filename = os.path.join(frame_dir, f"{saved:05d}.jpg")
                    cv2.imwrite(filename, frame)
                    saved += 1
                frame_count += 1
            cap.release()

        elif mode == "stream":
            # Linux cloud: streaming using FFmpeg + OpenCV
            import numpy as np

            ffmpeg_cmd = [
                "ffmpeg",
                "-i", "pipe:0",
                "-f", "image2pipe",
                "-pix_fmt", "bgr24",
                "-vcodec", "rawvideo",
                "-"
            ]

            ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=video_input.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # We need width/height; assume standard 1280x720
            width, height = 1280, 720
            frame_size = width * height * 3
            frame_count = 0

            while True:
                raw_frame = ffmpeg_process.stdout.read(frame_size)
                if not raw_frame:
                    break
                frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
                if frame_count % fps == 0:
                    filename = os.path.join(frame_dir, f"{saved:05d}.jpg")
                    cv2.imwrite(filename, frame)
                    saved += 1
                frame_count += 1

            ffmpeg_process.stdout.close()
            ffmpeg_process.stderr.close()
            video_input.stdout.close()

        # Save metadata
        metadata_path = os.path.join(self.base_dir, video_id, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({"frames": saved, "fps": fps}, f, indent=4)

        return saved