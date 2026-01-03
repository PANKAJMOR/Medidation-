

# import numpy as np
# from collections import defaultdict


# class LegMovement:
#     """
#     Segmented event-based leg movement detector.

#     Emits:
#         "START" → leg movement begins
#         "END"   → leg movement ends
#         None    → no state change
#     """

#     def __init__(
#         self,
#         ankle_thresh=12.0,          # pixels (primary)
#         knee_dist_thresh=25.0,      # pixels (secondary proxy)
#         hold_seconds=1.0,
#         fps=25,
#         stable_frames=25
#     ):
#         # -------------------------
#         # Config
#         # -------------------------
#         self.ankle_thresh = ankle_thresh
#         self.knee_dist_thresh = knee_dist_thresh
#         self.hold_frames = int(hold_seconds * fps)
#         self.stable_frames = stable_frames

#         # -------------------------
#         # YOLOv8 keypoints
#         # -------------------------
#         self.IDX_L_KNEE = 13
#         self.IDX_R_KNEE = 14
#         self.IDX_L_ANKLE = 15
#         self.IDX_R_ANKLE = 16

#         # -------------------------
#         # State
#         # -------------------------
#         self.baseline = defaultdict(lambda: None)
#         self.hold_counter = defaultdict(int)
#         self.stable_counter = defaultdict(int)
#         self.state = defaultdict(lambda: "STABLE")

#     # ==================================================
#     # UPDATE
#     # ==================================================
#     def update(self, person_id, keypoints):
#         """
#         Returns:
#             "START" | "END" | None
#         """
#         if keypoints is None:
#             return None

#         kp = np.asarray(keypoints)

#         required = [
#             self.IDX_L_KNEE, self.IDX_R_KNEE,
#             self.IDX_L_ANKLE, self.IDX_R_ANKLE
#         ]

#         for i in required:
#             if kp[i][0] <= 0 or kp[i][1] <= 0:
#                 return None

#         # -------------------------
#         # Current signals
#         # -------------------------
#         l_ankle = kp[self.IDX_L_ANKLE]
#         r_ankle = kp[self.IDX_R_ANKLE]

#         l_knee = kp[self.IDX_L_KNEE]
#         r_knee = kp[self.IDX_R_KNEE]

#         ankle_vec = np.concatenate([l_ankle, r_ankle])
#         knee_dist = np.linalg.norm(l_knee - r_knee)

#         # -------------------------
#         # Initialize baseline
#         # -------------------------
#         if self.baseline[person_id] is None:
#             self.baseline[person_id] = {
#                 "ankle": ankle_vec.copy(),
#                 "knee_dist": knee_dist
#             }
#             return None

#         base = self.baseline[person_id]

#         # -------------------------
#         # Primary: ankle movement
#         # -------------------------
#         la_d = np.linalg.norm(l_ankle - base["ankle"][0:2])
#         ra_d = np.linalg.norm(r_ankle - base["ankle"][2:4])
#         ankle_moved = la_d > self.ankle_thresh or ra_d > self.ankle_thresh

#         # -------------------------
#         # Secondary: knee distance change
#         # -------------------------
#         knee_dist_change = abs(knee_dist - base["knee_dist"])
#         knee_proxy_moved = knee_dist_change > self.knee_dist_thresh

#         # -------------------------
#         # Final decision
#         # -------------------------
#         leg_moving = ankle_moved or knee_proxy_moved

#         # -------------------------
#         # HOLD / STABILITY LOGIC
#         # -------------------------
#         signal = None

#         if leg_moving:
#             self.hold_counter[person_id] += 1
#             self.stable_counter[person_id] = 0
#         else:
#             self.hold_counter[person_id] = 0
#             self.stable_counter[person_id] += 1

#         # STABLE → MOVING
#         if (
#             self.state[person_id] == "STABLE"
#             and self.hold_counter[person_id] >= self.hold_frames
#         ):
#             self.state[person_id] = "MOVING"
#             signal = "START"
#             self.hold_counter[person_id] = 0

#         # MOVING → STABLE
#         elif (
#             self.state[person_id] == "MOVING"
#             and self.stable_counter[person_id] >= self.stable_frames
#         ):
#             self.state[person_id] = "STABLE"
#             signal = "END"
#             self.stable_counter[person_id] = 0
#             self.baseline[person_id] = {
#                 "ankle": ankle_vec.copy(),
#                 "knee_dist": knee_dist
#             }

#         return signal


import numpy as np
from collections import defaultdict


class LegMovement:
    """
    Segmented event-based leg movement detector.

    Emits:
        "START" → leg movement begins
        "END"   → leg movement ends
        None    → no state change
    """

    def __init__(
        self,
        ankle_thresh=12.0,          # pixels
        knee_dist_thresh=25.0,      # pixels (used as knee movement threshold)
        hold_seconds=1.0,
        fps=25,
        stable_frames=25
    ):
        # -------------------------
        # Config (UNCHANGED)
        # -------------------------
        self.ankle_thresh = ankle_thresh
        self.knee_thresh = knee_dist_thresh
        self.hold_frames = int(hold_seconds * fps)
        self.stable_frames = stable_frames

        # -------------------------
        # YOLOv8 keypoints
        # -------------------------
        self.IDX_L_KNEE = 13
        self.IDX_R_KNEE = 14
        self.IDX_L_ANKLE = 15
        self.IDX_R_ANKLE = 16

        # -------------------------
        # State
        # -------------------------
        self.prev_kpts = defaultdict(lambda: None)
        self.hold_counter = defaultdict(int)
        self.stable_counter = defaultdict(int)
        self.state = defaultdict(lambda: "STABLE")

    # ==================================================
    # UPDATE (INTERFACE UNCHANGED)
    # ==================================================
    def update(self, person_id, keypoints):
        """
        Returns:
            "START" | "END" | None
        """
        if keypoints is None:
            return None

        kp = np.asarray(keypoints)

        required = [
            self.IDX_L_KNEE, self.IDX_R_KNEE,
            self.IDX_L_ANKLE, self.IDX_R_ANKLE
        ]

        for i in required:
            if kp[i][0] <= 0 or kp[i][1] <= 0:
                return None

        # First frame
        if self.prev_kpts[person_id] is None:
            self.prev_kpts[person_id] = kp
            return None

        prev = self.prev_kpts[person_id]

        # -------------------------
        # HARD MOVEMENT CHECK
        # -------------------------
        la = np.linalg.norm(kp[self.IDX_L_ANKLE] - prev[self.IDX_L_ANKLE])
        ra = np.linalg.norm(kp[self.IDX_R_ANKLE] - prev[self.IDX_R_ANKLE])
        lk = np.linalg.norm(kp[self.IDX_L_KNEE] - prev[self.IDX_L_KNEE])
        rk = np.linalg.norm(kp[self.IDX_R_KNEE] - prev[self.IDX_R_KNEE])

        leg_moving = (
            la > self.ankle_thresh or
            ra > self.ankle_thresh or
            lk > self.knee_thresh or
            rk > self.knee_thresh
        )

        signal = None

        # -------------------------
        # HOLD / STABILITY LOGIC
        # -------------------------
        if leg_moving:
            self.hold_counter[person_id] += 1
            self.stable_counter[person_id] = 0
        else:
            self.hold_counter[person_id] = 0
            self.stable_counter[person_id] += 1

        # STABLE → MOVING
        if (
            self.state[person_id] == "STABLE"
            and self.hold_counter[person_id] >= self.hold_frames
        ):
            self.state[person_id] = "MOVING"
            self.hold_counter[person_id] = 0
            signal = "START"

        # MOVING → STABLE
        elif (
            self.state[person_id] == "MOVING"
            and self.stable_counter[person_id] >= self.stable_frames
        ):
            self.state[person_id] = "STABLE"
            self.stable_counter[person_id] = 0
            signal = "END"

        self.prev_kpts[person_id] = kp
        return signal

