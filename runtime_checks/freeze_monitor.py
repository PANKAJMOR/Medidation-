import cv2
import hashlib
from collections import defaultdict


class RuntimeFreezeMonitor:
    def __init__(self, freeze_seconds=15 * 60, fps=1):
        """
        freeze_seconds : allowed freeze duration (default 15 min)
        fps            : processing fps (your pipeline = 1 FPS)
        """
        self.max_same_frames = freeze_seconds * fps
        self.last_hash = None
        self.same_counter = 0

    def _frame_hash(self, frame):
        """
        Robust perceptual hash using resized grayscale frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64))
        return hashlib.md5(small.tobytes()).hexdigest()

    def update(self, frame):
        """
        Returns:
        - None → OK
        - dict → FREEZE ERROR
        """
        curr_hash = self._frame_hash(frame)

        if self.last_hash is None:
            self.last_hash = curr_hash
            return None

        if curr_hash == self.last_hash:
            self.same_counter += 1
        else:
            self.same_counter = 0
            self.last_hash = curr_hash

        if self.same_counter >= self.max_same_frames:
            return {
                "code": "VIDEO_DISCONTINUITY",
                "message": (
                    f"Video frozen for more than "
                    f"{self.max_same_frames} seconds"
                )
            }

        return None
