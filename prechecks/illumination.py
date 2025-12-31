import cv2
import numpy as np
from prechecks.base import BasePrecheck, PrecheckResult

class IlluminationCheck(BasePrecheck):

    def __init__(self, min_brightness=40):
        self.min_brightness = min_brightness

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return PrecheckResult(False, "FRAME_READ_FAIL", "Cannot read frame")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if brightness < self.min_brightness:
            return PrecheckResult(
                False,
                "POOR_VIDEO_QUALITY",
                f"Brightness too low ({brightness:.1f})"
            )

        return PrecheckResult(True)
