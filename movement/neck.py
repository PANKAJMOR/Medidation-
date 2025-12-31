import numpy as np
import math
from collections import defaultdict


class NeckMovement:
    """
    Robust neck movement detector.
    - Horizontal: yaw angle
    - Vertical: nose Y displacement
    """

    def __init__(
        self,
        yaw_thresh=7.0,           # degrees
        nose_y_thresh=8.0,        # pixels
        hold_seconds=0.35,
        fps=25,
        min_still_frames=6
    ):
        self.yaw_thresh = yaw_thresh
        self.nose_y_thresh = nose_y_thresh
        self.hold_frames = int(hold_seconds * fps)
        self.min_still_frames = min_still_frames

        # YOLOv8 indices
        self.IDX_NOSE = 0
        self.IDX_L_SHOULDER = 5
        self.IDX_R_SHOULDER = 6

        # State
        self.prev_yaw = defaultdict(lambda: None)
        self.prev_nose_y = defaultdict(lambda: None)

        self.hold_counter = defaultdict(int)
        self.still_counter = defaultdict(int)

        self.state = defaultdict(lambda: "STILL")
        self.count = defaultdict(int)

    # --------------------------------------------------
    def update(self, person_id, keypoints):
        if keypoints is None:
            return

        kp = np.asarray(keypoints)

        nose = kp[self.IDX_NOSE]
        ls = kp[self.IDX_L_SHOULDER]
        rs = kp[self.IDX_R_SHOULDER]

        if np.any(nose <= 0) or np.any(ls <= 0) or np.any(rs <= 0):
            return

        mid = (ls + rs) / 2
        v = nose - mid

        # Horizontal (yaw)
        yaw = math.degrees(math.atan2(v[0], v[1]))

        # Vertical (nose translation)
        nose_y = nose[1]

        if self.prev_yaw[person_id] is None:
            self.prev_yaw[person_id] = yaw
            self.prev_nose_y[person_id] = nose_y
            return

        dyaw = abs(yaw - self.prev_yaw[person_id])
        dny = abs(nose_y - self.prev_nose_y[person_id])

        neck_moving = (
            dyaw > self.yaw_thresh or
            dny > self.nose_y_thresh
        )

        # Hold logic
        if neck_moving:
            self.hold_counter[person_id] += 1
            self.still_counter[person_id] = 0
        else:
            self.hold_counter[person_id] = 0
            self.still_counter[person_id] += 1

        # Count once per movement
        if (
            self.state[person_id] == "STILL"
            and self.hold_counter[person_id] >= self.hold_frames
        ):
            self.state[person_id] = "MOVING"
            self.count[person_id] += 1

        if (
            self.state[person_id] == "MOVING"
            and self.still_counter[person_id] >= self.min_still_frames
        ):
            self.state[person_id] = "STILL"

        self.prev_yaw[person_id] = yaw
        self.prev_nose_y[person_id] = nose_y

    # --------------------------------------------------
    def get_count(self, person_id):
        return self.count[person_id]
