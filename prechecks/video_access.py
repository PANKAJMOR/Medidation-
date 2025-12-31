import cv2
import os
from prechecks.base import BasePrecheck, PrecheckResult

class VideoAccessCheck(BasePrecheck):

    def run(self, video_path):
        if os.path.exists(video_path):
            PrecheckResult(ok=True)  # 🔑 skip link logic entirely
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return PrecheckResult(
                ok=False,
                error_code="VIDEO_NOT_ACCESSIBLE",
                message="Video file/link could not be opened"
            )

        cap.release()
        return PrecheckResult(ok=True)
