# import cv2
# import numpy as np
# import mediapipe as mp
# from collections import defaultdict
# import math


# class FaceNeckMovement:
#     def __init__(
#         self,
#         yaw_delta_thresh=6.0,
#         pitch_delta_thresh=6.0,
#         hold_frames=3,
#         cooldown_seconds=0.6,
#         fps=25
#     ):
#         self.yaw_delta_thresh = yaw_delta_thresh
#         self.pitch_delta_thresh = pitch_delta_thresh
#         self.hold_frames = hold_frames
#         self.cooldown_frames = int(cooldown_seconds * fps)

#         self.mp_face = mp.solutions.face_mesh.FaceMesh(
#             static_image_mode=False,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )

#         # State
#         self.prev_angles = {}
#         self.hold_counter = defaultdict(int)
#         self.cooldown_counter = defaultdict(int)
#         self.count = defaultdict(int)

#     # --------------------------------------------------
#     def update(self, person_id, frame, face_bbox, draw=False):
#         x1, y1, x2, y2 = face_bbox
#         h, w, _ = frame.shape

#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(w, x2), min(h, y2)

#         face = frame[y1:y2, x1:x2]
#         if face.size == 0:
#             return

#         face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#         results = self.mp_face.process(face_rgb)

#         if not results.multi_face_landmarks:
#             return

#         lm = results.multi_face_landmarks[0].landmark

#         nose = np.array([lm[1].x, lm[1].y])
#         left_eye = np.array([lm[33].x, lm[33].y])
#         right_eye = np.array([lm[263].x, lm[263].y])
#         eye_mid = (left_eye + right_eye) / 2

#         v = nose - eye_mid
#         yaw = math.degrees(math.atan2(v[0], v[1]))
#         pitch = math.degrees(math.atan2(-v[1], abs(v[0]) + 1e-6))

#         # Draw landmarks (debug)
#         if draw:
#             for idx in [1, 33, 263]:
#                 px = int(lm[idx].x * (x2 - x1)) + x1
#                 py = int(lm[idx].y * (y2 - y1)) + y1
#                 cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

#         # Init
#         if person_id not in self.prev_angles:
#             self.prev_angles[person_id] = (yaw, pitch)
#             return

#         # Cooldown active → skip counting
#         if self.cooldown_counter[person_id] > 0:
#             self.cooldown_counter[person_id] -= 1
#             self.prev_angles[person_id] = (yaw, pitch)
#             return

#         prev_yaw, prev_pitch = self.prev_angles[person_id]
#         dyaw = abs(yaw - prev_yaw)
#         dpitch = abs(pitch - prev_pitch)

#         moving = (
#             dyaw > self.yaw_delta_thresh or
#             dpitch > self.pitch_delta_thresh
#         )

#         if moving:
#             self.hold_counter[person_id] += 1
#         else:
#             self.hold_counter[person_id] = 0

#         # Count ONCE per motion
#         if self.hold_counter[person_id] == self.hold_frames:
#             self.count[person_id] += 1
#             self.cooldown_counter[person_id] = self.cooldown_frames
#             self.hold_counter[person_id] = 0

#         self.prev_angles[person_id] = (yaw, pitch)

#     # --------------------------------------------------
#     def get_count(self, person_id):
#         return self.count[person_id]


import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict
import math


class FaceNeckMovement:
    def __init__(
        self,
        yaw_delta_thresh=6.0,
        pitch_delta_thresh=6.0,
        hold_frames=3,
        cooldown_seconds=0.6,
        fps=25,
        min_still_frames=4
    ):
        self.yaw_delta_thresh = yaw_delta_thresh
        self.pitch_delta_thresh = pitch_delta_thresh
        self.hold_frames = hold_frames
        self.cooldown_frames = int(cooldown_seconds * fps)
        self.min_still_frames = min_still_frames

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # -------------------------
        # State (per person)
        # -------------------------
        self.prev_angles = {}
        self.hold_counter = defaultdict(int)
        self.still_counter = defaultdict(int)
        self.cooldown_counter = defaultdict(int)
        self.state = defaultdict(lambda: "STILL")

    # --------------------------------------------------
    def update(self, person_id, frame, face_bbox, draw=False):
        """
        Returns:
            "START" | "END" | None
        """

        x1, y1, x2, y2 = face_bbox
        h, w, _ = frame.shape

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(face_rgb)

        if not results.multi_face_landmarks:
            self.still_counter[person_id] += 1
            self.hold_counter[person_id] = 0
            return None

        lm = results.multi_face_landmarks[0].landmark

        nose = np.array([lm[1].x, lm[1].y])
        left_eye = np.array([lm[33].x, lm[33].y])
        right_eye = np.array([lm[263].x, lm[263].y])
        eye_mid = (left_eye + right_eye) / 2

        v = nose - eye_mid
        yaw = math.degrees(math.atan2(v[0], v[1]))
        pitch = math.degrees(math.atan2(-v[1], abs(v[0]) + 1e-6))

        # Debug draw
        if draw:
            for idx in [1, 33, 263]:
                px = int(lm[idx].x * (x2 - x1)) + x1
                py = int(lm[idx].y * (y2 - y1)) + y1
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

        # Init
        if person_id not in self.prev_angles:
            self.prev_angles[person_id] = (yaw, pitch)
            return None

        # Cooldown active
        if self.cooldown_counter[person_id] > 0:
            self.cooldown_counter[person_id] -= 1
            self.prev_angles[person_id] = (yaw, pitch)
            return None

        prev_yaw, prev_pitch = self.prev_angles[person_id]
        dyaw = abs(yaw - prev_yaw)
        dpitch = abs(pitch - prev_pitch)

        moving = (
            dyaw > self.yaw_delta_thresh or
            dpitch > self.pitch_delta_thresh
        )

        # -------------------------
        # Counters
        # -------------------------
        if moving:
            self.hold_counter[person_id] += 1
            self.still_counter[person_id] = 0
        else:
            self.hold_counter[person_id] = 0
            self.still_counter[person_id] += 1

        # -------------------------
        # STILL → MOVING
        # -------------------------
        if (
            self.state[person_id] == "STILL"
            and self.hold_counter[person_id] >= self.hold_frames
        ):
            self.state[person_id] = "MOVING"
            self.cooldown_counter[person_id] = self.cooldown_frames
            self.hold_counter[person_id] = 0
            return "START"

        # -------------------------
        # MOVING → STILL
        # -------------------------
        if (
            self.state[person_id] == "MOVING"
            and self.still_counter[person_id] >= self.min_still_frames
        ):
            self.state[person_id] = "STILL"
            return "END"

        self.prev_angles[person_id] = (yaw, pitch)
        return None
    
    # --------------------------------------------------
   
