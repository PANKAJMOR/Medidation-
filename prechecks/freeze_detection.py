import cv2
import numpy as np
from prechecks.base import BasePrecheck, PrecheckResult

class FreezeCheck(BasePrecheck):

    def __init__(
        self,
        sample_frames=60,
        diff_thresh=1.0,
        identical_ratio_thresh=1.0
    ):
        self.sample_frames = sample_frames
        self.diff_thresh = diff_thresh
        self.identical_ratio_thresh = identical_ratio_thresh

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)

        prev_gray = None
        identical_frames = 0
        total = 0

        for _ in range(self.sample_frames):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray))

                # STRICT identical check (not just low motion)
                if diff < self.diff_thresh:
                    identical_frames += 1

                total += 1

            prev_gray = gray

        cap.release()

        if total == 0:
            return PrecheckResult(
                False,
                "VIDEO_DISCONTINUITY",
                "Unable to read video frames"
            )

        identical_ratio = identical_frames / total

        # 🔑 This is the key decision
        if identical_ratio > self.identical_ratio_thresh:
            return PrecheckResult(
                False,
                "VIDEO_DISCONTINUITY",
                f"Frames repeated too often ({identical_ratio:.2f})"
            )

        return PrecheckResult(True)
