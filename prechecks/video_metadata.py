import cv2
from prechecks.base import BasePrecheck, PrecheckResult
from datetime import datetime
import json


class VideoMetadataCheck(BasePrecheck):

    def __init__(self, min_duration_sec=2.5 * 3600):
        self.min_duration_sec = min_duration_sec


    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frames / max(fps, 1)

        cap.release()

        if duration < self.min_duration_sec:
            return PrecheckResult(
                ok=False,
                error_code="VIDEO_TOO_SHORT",
                message=f"Video duration {duration:.1f}s < required"
            )

        return PrecheckResult(ok=True)
