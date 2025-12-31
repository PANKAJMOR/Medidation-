import numpy as np
from collections import defaultdict
import math


class MovementCounter:
    def __init__(
        self,
        neck_yaw_thresh=12.0,     # degrees
        neck_pitch_thresh=10.0,   # degrees
        neck_hold_seconds=0.5,
        fps=25,
        min_still_frames=6
    ):
        # -------------------------
        # Neck config
        # -------------------------
        self.neck_yaw_thresh = neck_yaw_thresh
        self.neck_pitch_thresh = neck_pitch_thresh
        self.neck_hold_frames = int(neck_hold_seconds * fps)

        # -------------------------
        # Pose indices (YOLOv8)
        # -------------------------
        self.idx_nose = 0
        self.idx_l_shoulder = 5
        self.idx_r_shoulder = 6

        # -------------------------
        # State
        # -------------------------
        self.neck_baseline = defaultdict(lambda: None)
        self.neck_hold_counter = defaultdict(int)
        self.neck_still_counter = defaultdict(int)

        self.neck_state = defaultdict(lambda: "STILL")
        self.neck_count = defaultdict(int)

        self.min_still_frames = min_still_frames

    # ==================================================
    # MAIN UPDATE
    # ==================================================
    def update(self, person_id, keypoints):
        if keypoints is None:
            return

        kp = np.asarray(keypoints)

        nose = kp[self.idx_nose]
        ls = kp[self.idx_l_shoulder]
        rs = kp[self.idx_r_shoulder]

        # Skip if keypoints are invalid
        if np.any(nose <= 0) or np.any(ls <= 0) or np.any(rs <= 0):
            return

        # Shoulder midpoint
        mid = (ls + rs) / 2

        # Vector shoulders -> nose
        v = nose - mid

        # -------------------------
        # Angles
        # -------------------------
        yaw = math.degrees(math.atan2(v[0], v[1]))      # left/right
        pitch = math.degrees(math.atan2(-v[1], abs(v[0]) + 1e-6))  # up/down

        # Initialize baseline
        if self.neck_baseline[person_id] is None:
            self.neck_baseline[person_id] = (yaw, pitch)
            return

        base_yaw, base_pitch = self.neck_baseline[person_id]

        dyaw = abs(yaw - base_yaw)
        dpitch = abs(pitch - base_pitch)

        neck_moving = (
            dyaw > self.neck_yaw_thresh or
            dpitch > self.neck_pitch_thresh
        )

        # -------------------------
        # HOLD-TIME LOGIC
        # -------------------------
        if neck_moving:
            self.neck_hold_counter[person_id] += 1
            self.neck_still_counter[person_id] = 0
        else:
            self.neck_hold_counter[person_id] = 0
            self.neck_still_counter[person_id] += 1

        # STILL → MOVING (COUNT)
        if (
            self.neck_state[person_id] == "STILL"
            and self.neck_hold_counter[person_id] >= self.neck_hold_frames
        ):
            self.neck_state[person_id] = "MOVING"
            self.neck_count[person_id] += 1

        # MOVING → STILL (RESET BASELINE)
        if (
            self.neck_state[person_id] == "MOVING"
            and self.neck_still_counter[person_id] >= self.min_still_frames
        ):
            self.neck_state[person_id] = "STILL"
            self.neck_baseline[person_id] = (yaw, pitch)

    # ==================================================
    # PUBLIC
    # ==================================================
    def get_counts(self, person_id):
        return {
            "neck": self.neck_count[person_id]
        }
