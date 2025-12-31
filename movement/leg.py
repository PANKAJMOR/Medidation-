# import numpy as np
# from collections import defaultdict


# class LegMovement:
#     """
#     Segmented event-based leg movement counter.

#     Leg movement is counted when:
#     1) Ankles move significantly (feet / toe motion), OR
#     2) Distance between knees changes significantly (lap/thigh reposition)

#     Logic:
#     STABLE → MOVING → STABLE
#                   ↑
#                count +1
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
#         self.count = defaultdict(int)

#     # ==================================================
#     # UPDATE
#     # ==================================================
#     def update(self, person_id, keypoints):
#         if keypoints is None:
#             return

#         kp = np.asarray(keypoints)

#         required = [
#             self.IDX_L_KNEE, self.IDX_R_KNEE,
#             self.IDX_L_ANKLE, self.IDX_R_ANKLE
#         ]

#         for i in required:
#             if kp[i][0] <= 0 or kp[i][1] <= 0:
#                 return

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
#             return

#         base = self.baseline[person_id]

#         # -------------------------
#         # Ankle movement (primary)
#         # -------------------------
#         la_d = np.linalg.norm(l_ankle - base["ankle"][0:2])
#         ra_d = np.linalg.norm(r_ankle - base["ankle"][2:4])

#         ankle_moved = la_d > self.ankle_thresh or ra_d > self.ankle_thresh

#         # -------------------------
#         # Knee distance change (secondary)
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
#         if leg_moving:
#             self.hold_counter[person_id] += 1
#             self.stable_counter[person_id] = 0
#         else:
#             self.hold_counter[person_id] = 0
#             self.stable_counter[person_id] += 1

#         # -------------------------
#         # STABLE → MOVING (COUNT)
#         # -------------------------
#         if (
#             self.state[person_id] == "STABLE"
#             and self.hold_counter[person_id] >= self.hold_frames
#         ):
#             self.state[person_id] = "MOVING"
#             self.count[person_id] += 1

#         # -------------------------
#         # MOVING → STABLE (RESET BASELINE)
#         # -------------------------
#         if (
#             self.state[person_id] == "MOVING"
#             and self.stable_counter[person_id] >= self.stable_frames
#         ):
#             self.state[person_id] = "STABLE"
#             self.baseline[person_id] = {
#                 "ankle": ankle_vec.copy(),
#                 "knee_dist": knee_dist
#             }

#     # ==================================================
#     # PUBLIC
#     # ==================================================
#     def get_count(self, person_id):
#         return self.count[person_id]


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
        ankle_thresh=12.0,          # pixels (primary)
        knee_dist_thresh=25.0,      # pixels (secondary proxy)
        hold_seconds=1.0,
        fps=25,
        stable_frames=25
    ):
        # -------------------------
        # Config
        # -------------------------
        self.ankle_thresh = ankle_thresh
        self.knee_dist_thresh = knee_dist_thresh
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
        self.baseline = defaultdict(lambda: None)
        self.hold_counter = defaultdict(int)
        self.stable_counter = defaultdict(int)
        self.state = defaultdict(lambda: "STABLE")

    # ==================================================
    # UPDATE
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

        # -------------------------
        # Current signals
        # -------------------------
        l_ankle = kp[self.IDX_L_ANKLE]
        r_ankle = kp[self.IDX_R_ANKLE]

        l_knee = kp[self.IDX_L_KNEE]
        r_knee = kp[self.IDX_R_KNEE]

        ankle_vec = np.concatenate([l_ankle, r_ankle])
        knee_dist = np.linalg.norm(l_knee - r_knee)

        # -------------------------
        # Initialize baseline
        # -------------------------
        if self.baseline[person_id] is None:
            self.baseline[person_id] = {
                "ankle": ankle_vec.copy(),
                "knee_dist": knee_dist
            }
            return None

        base = self.baseline[person_id]

        # -------------------------
        # Primary: ankle movement
        # -------------------------
        la_d = np.linalg.norm(l_ankle - base["ankle"][0:2])
        ra_d = np.linalg.norm(r_ankle - base["ankle"][2:4])
        ankle_moved = la_d > self.ankle_thresh or ra_d > self.ankle_thresh

        # -------------------------
        # Secondary: knee distance change
        # -------------------------
        knee_dist_change = abs(knee_dist - base["knee_dist"])
        knee_proxy_moved = knee_dist_change > self.knee_dist_thresh

        # -------------------------
        # Final decision
        # -------------------------
        leg_moving = ankle_moved or knee_proxy_moved

        # -------------------------
        # HOLD / STABILITY LOGIC
        # -------------------------
        signal = None

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
            signal = "START"
            self.hold_counter[person_id] = 0

        # MOVING → STABLE
        elif (
            self.state[person_id] == "MOVING"
            and self.stable_counter[person_id] >= self.stable_frames
        ):
            self.state[person_id] = "STABLE"
            signal = "END"
            self.stable_counter[person_id] = 0
            self.baseline[person_id] = {
                "ankle": ankle_vec.copy(),
                "knee_dist": knee_dist
            }

        return signal
