from ingestion.video_ingestion import VideoIngestion
from pipeline.analyze_video import analyze_video
import os
import time
import cv2


def wait_for_video_ready(path, timeout=300):
    """
    Wait until ffprobe can read duration and streams.
    """
    import subprocess
    import time

    start = time.time()

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]

    while True:
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True
            )

            if result.returncode == 0:
                output = result.stdout.strip().splitlines()
                if len(output) >= 4:  # width, height, codec, duration
                    return  # ✅ READY

        except Exception:
            pass

        if time.time() - start > timeout:
            raise TimeoutError(f"Video never became readable: {path}")

        time.sleep(1)


def run_analysis(input_source):
    ingestor = VideoIngestion()

    try:
        local_video_path, video_id, meta = ingestor.ingest(input_source)
        local_video_path = os.path.abspath(local_video_path)

        print("Ingested video path:", local_video_path)

    except Exception as e:
        return {
            "status": "FAILED",
            "errors": [{
                "code": "VIDEO_INGESTION_FAILED",
                "message": str(e)
            }]
        }

    return analyze_video(local_video_path)
