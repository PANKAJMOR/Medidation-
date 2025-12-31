# import numpy as np
# from collections import defaultdict


# class ArmMovement:
#     def __init__(
#         self,
#         wrist_thresh=6.0,      # pixels
#         elbow_thresh=8.0,      # pixels
#         hold_seconds=0.4,
#         fps=25,
#         lap_margin=20,
#         min_still_frames=6
#     ):
#         # -------------------------
#         # Config
#         # -------------------------
#         self.wrist_thresh = wrist_thresh
#         self.elbow_thresh = elbow_thresh
#         self.hold_frames = int(hold_seconds * fps)
#         self.lap_margin = lap_margin
#         self.min_still_frames = min_still_frames

#         # -------------------------
#         # YOLOv8 keypoint indices
#         # -------------------------
#         self.IDX_L_WRIST = 9
#         self.IDX_R_WRIST = 10
#         self.IDX_L_ELBOW = 7
#         self.IDX_R_ELBOW = 8
#         self.IDX_L_HIP = 11
#         self.IDX_R_HIP = 12

#         # -------------------------
#         # State
#         # -------------------------
#         self.prev_kpts = defaultdict(lambda: None)

#         self.hold_counter = defaultdict(int)
#         self.still_counter = defaultdict(int)

#         self.state = defaultdict(lambda: "STILL")
#         self.count = defaultdict(int)

#     # ==================================================
#     # UPDATE
#     # ==================================================
#     def update(self, person_id, keypoints):
#         if keypoints is None:
#             return

#         kp = np.asarray(keypoints)

#         # Required keypoints
#         required = [
#             self.IDX_L_WRIST, self.IDX_R_WRIST,
#             self.IDX_L_ELBOW, self.IDX_R_ELBOW,
#             self.IDX_L_HIP, self.IDX_R_HIP
#         ]

#         for i in required:
#             if kp[i][0] <= 0 or kp[i][1] <= 0:
#                 return

#         # First frame
#         if self.prev_kpts[person_id] is None:
#             self.prev_kpts[person_id] = kp
#             return

#         prev = self.prev_kpts[person_id]

#         # -------------------------
#         # Lap suppression
#         # -------------------------
#         hip_y = (kp[self.IDX_L_HIP][1] + kp[self.IDX_R_HIP][1]) / 2

#         lw_on_lap = kp[self.IDX_L_WRIST][1] > hip_y - self.lap_margin
#         rw_on_lap = kp[self.IDX_R_WRIST][1] > hip_y - self.lap_margin

#         # -------------------------
#         # Displacements
#         # -------------------------
#         lw_dist = np.linalg.norm(
#             kp[self.IDX_L_WRIST] - prev[self.IDX_L_WRIST]
#         )
#         rw_dist = np.linalg.norm(
#             kp[self.IDX_R_WRIST] - prev[self.IDX_R_WRIST]
#         )

#         le_dist = np.linalg.norm(
#             kp[self.IDX_L_ELBOW] - prev[self.IDX_L_ELBOW]
#         )
#         re_dist = np.linalg.norm(
#             kp[self.IDX_R_ELBOW] - prev[self.IDX_R_ELBOW]
#         )

#         wrist_move = lw_dist > self.wrist_thresh or rw_dist > self.wrist_thresh
#         elbow_move = le_dist > self.elbow_thresh or re_dist > self.elbow_thresh

#         # -------------------------
#         # Final arm decision
#         # -------------------------
#         arm_moving = (
#             (wrist_move or elbow_move) and
#             not (lw_on_lap and rw_on_lap)
#         )

#         # -------------------------
#         # HOLD-TIME LOGIC
#         # -------------------------
#         if arm_moving:
#             self.hold_counter[person_id] += 1
#             self.still_counter[person_id] = 0
#         else:
#             self.hold_counter[person_id] = 0
#             self.still_counter[person_id] += 1

#         # STILL → MOVING
#         if (
#             self.state[person_id] == "STILL"
#             and self.hold_counter[person_id] >= self.hold_frames
#         ):
#             self.state[person_id] = "MOVING"
#             self.count[person_id] += 1

#         # MOVING → STILL
#         if (
#             self.state[person_id] == "MOVING"
#             and self.still_counter[person_id] >= self.min_still_frames
#         ):
#             self.state[person_id] = "STILL"

#         # Save for next frame
#         self.prev_kpts[person_id] = kp

#     # ==================================================
#     # PUBLIC
#     # ==================================================
#     def get_count(self, person_id):
#         return self.count[person_id]


import numpy as np
from collections import defaultdict


class ArmMovement:
    def __init__(
        self,
        wrist_thresh=6.0,      # pixels
        elbow_thresh=8.0,      # pixels
        hold_seconds=0.4,
        fps=25,
        lap_margin=20,
        min_still_frames=6
    ):
        # -------------------------
        # Config
        # -------------------------
        self.wrist_thresh = wrist_thresh
        self.elbow_thresh = elbow_thresh
        self.hold_frames = int(hold_seconds * fps)
        self.lap_margin = lap_margin
        self.min_still_frames = min_still_frames

        # -------------------------
        # YOLOv8 keypoint indices
        # -------------------------
        self.IDX_L_WRIST = 9
        self.IDX_R_WRIST = 10
        self.IDX_L_ELBOW = 7
        self.IDX_R_ELBOW = 8
        self.IDX_L_HIP = 11
        self.IDX_R_HIP = 12

        # -------------------------
        # State
        # -------------------------
        self.prev_kpts = defaultdict(lambda: None)
        self.hold_counter = defaultdict(int)
        self.still_counter = defaultdict(int)
        self.state = defaultdict(lambda: "STILL")

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
            self.IDX_L_WRIST, self.IDX_R_WRIST,
            self.IDX_L_ELBOW, self.IDX_R_ELBOW,
            self.IDX_L_HIP, self.IDX_R_HIP
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
        # Lap suppression
        # -------------------------
        hip_y = (kp[self.IDX_L_HIP][1] + kp[self.IDX_R_HIP][1]) / 2

        lw_on_lap = kp[self.IDX_L_WRIST][1] > hip_y - self.lap_margin
        rw_on_lap = kp[self.IDX_R_WRIST][1] > hip_y - self.lap_margin

        # -------------------------
        # Displacements
        # -------------------------
        lw_dist = np.linalg.norm(kp[self.IDX_L_WRIST] - prev[self.IDX_L_WRIST])
        rw_dist = np.linalg.norm(kp[self.IDX_R_WRIST] - prev[self.IDX_R_WRIST])
        le_dist = np.linalg.norm(kp[self.IDX_L_ELBOW] - prev[self.IDX_L_ELBOW])
        re_dist = np.linalg.norm(kp[self.IDX_R_ELBOW] - prev[self.IDX_R_ELBOW])

        wrist_move = lw_dist > self.wrist_thresh or rw_dist > self.wrist_thresh
        elbow_move = le_dist > self.elbow_thresh or re_dist > self.elbow_thresh

        arm_moving = (
            (wrist_move or elbow_move)
            and not (lw_on_lap and rw_on_lap)
        )

        # -------------------------
        # HOLD / STILL LOGIC
        # -------------------------
        signal = None

        if arm_moving:
            self.hold_counter[person_id] += 1
            self.still_counter[person_id] = 0
        else:
            self.hold_counter[person_id] = 0
            self.still_counter[person_id] += 1

        # STILL → MOVING
        if (
            self.state[person_id] == "STILL"
            and self.hold_counter[person_id] >= self.hold_frames
        ):
            self.state[person_id] = "MOVING"
            signal = "START"
            self.hold_counter[person_id] = 0

        # MOVING → STILL
        elif (
            self.state[person_id] == "MOVING"
            and self.still_counter[person_id] >= self.min_still_frames
        ):
            self.state[person_id] = "STILL"
            signal = "END"
            self.still_counter[person_id] = 0

        self.prev_kpts[person_id] = kp
        return signal
