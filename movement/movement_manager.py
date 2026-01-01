# from collections import defaultdict
# import time

# from movement.neck_face import FaceNeckMovement
# from movement.arm import ArmMovement
# from movement.leg import LegMovement


# class MovementManager:
#     def __init__(self, fps=25):
#         self.fps = fps

#         # -------------------------
#         # Sub-modules
#         # -------------------------
#         self.neck = FaceNeckMovement(
#             yaw_delta_thresh=9.0,      # degrees CHANGE
#             pitch_delta_thresh=8.0,
#             hold_frames=3,
#             cooldown_seconds=2.0,
#             fps=25
#         )

#         self.arm = ArmMovement(
#             wrist_thresh=4,
#             elbow_thresh=6,
#             hold_seconds=0.6,
#             fps=25,
#             lap_margin=20
#         )

#         self.leg = LegMovement(
#             ankle_thresh=20,
#             knee_dist_thresh=30,
#             hold_seconds=1.5,
#             fps=25,
#             stable_frames=25
#         )

#         # -------------------------
#         # Unified counters
#         # -------------------------
#         self.counts = defaultdict(lambda: {
#             "neck": 0,
#             "arm": 0,
#             "leg": 0
#         })

#     # --------------------------------------------------
#     def update(
#         self,
#         person_id,
#         frame,
#         keypoints,
#         face_bbox,
#         timestamp=None,
#         draw_debug=False
#     ):
#         """
#         frame       : full frame (for MediaPipe neck)
#         keypoints   : YOLO pose keypoints (for arm & leg)
#         face_bbox   : face crop bbox (for neck)
#         timestamp   : optional (for logging later)
#         """

#         # -------------------------
#         # Neck
#         # -------------------------
#         prev_neck = self.neck.get_count(person_id)
#         self.neck.update(person_id, frame, face_bbox, draw=draw_debug)
#         new_neck = self.neck.get_count(person_id)

#         if new_neck > prev_neck:
#             self.counts[person_id]["neck"] += 1

#         # -------------------------
#         # Arm
#         # -------------------------
#         prev_arm = self.arm.get_count(person_id)
#         self.arm.update(person_id, keypoints)
#         new_arm = self.arm.get_count(person_id)

#         if new_arm > prev_arm:
#             self.counts[person_id]["arm"] += 1

#         # -------------------------
#         # Leg
#         # -------------------------
#         prev_leg = self.leg.get_count(person_id)
#         self.leg.update(person_id, keypoints)
#         new_leg = self.leg.get_count(person_id)

#         if new_leg > prev_leg:
#             self.counts[person_id]["leg"] += 1

#     # --------------------------------------------------
#     def get_counts(self, person_id):
#         return self.counts[person_id]

#     # --------------------------------------------------
#     def get_all_counts(self):
#         return dict(self.counts)


from collections import defaultdict
from movement.neck_face import FaceNeckMovement
from movement.arm import ArmMovement
from movement.leg import LegMovement

class MovementManager:
    def __init__(self, fps=25):
        self.fps = fps
        self.initialized = False
        self.discontinued_once = set()


        # -------------------------
        # Sub-modules
        # -------------------------
        self.neck = FaceNeckMovement(
            yaw_delta_thresh=6.0,
            pitch_delta_thresh=5.0,
            hold_frames=3,
            cooldown_seconds=2.0,
            fps=fps
        )

        self.arm = ArmMovement(
            wrist_thresh=4,
            elbow_thresh=6,
            hold_seconds=0.6,
            fps=fps,
            lap_margin=20
        )

        self.leg = LegMovement(
            ankle_thresh=10,
            knee_dist_thresh=20,
            hold_seconds=1.5,
            fps=fps,
            stable_frames=25
        )

        # -------------------------
        # Counts
        # -------------------------
        self.counts = defaultdict(lambda: {
            "neck": 0,
            "arm": 0,
            "leg": 0,
            "discontinued": False
        })

        # -------------------------
        # Timestamp storage (RAW SECONDS)
        # -------------------------
        self.timestamps = defaultdict(lambda: {
            "neck": [],
            "arm": [],
            "leg": [],
            "discontinuity": []
        })

        #self.discontinuity_timestamps = defaultdict(list) # Stores Particant Discontinued timestamp


        # Active movements
        self.active = defaultdict(lambda: {
            "neck": None,
            "arm": None,
            "leg": None
        })

    # --------------------------------------------------
    def update(
        self,
        person_id,
        frame,
        keypoints,
        face_bbox,
        frame_sec,
        draw_debug=False
    ):
        """
        frame_sec:
            Seconds since ANALYSIS WINDOW START (float)
        """

        # -------------------------
        # NECK
        # -------------------------
        neck_event = self.neck.update(
            person_id,
            frame,
            face_bbox,
            draw=draw_debug
        )

        if neck_event == "START":
            self.counts[person_id]["neck"] += 1
            self.active[person_id]["neck"] = frame_sec

        elif neck_event == "END":
            start_ts = self.active[person_id]["neck"]
            if start_ts is not None:
                self.timestamps[person_id]["neck"].append({
                    "start": start_ts,
                    "end": frame_sec
                })
            self.active[person_id]["neck"] = None

        # -------------------------
        # ARM
        # -------------------------
        arm_event = self.arm.update(person_id, keypoints)

        if arm_event == "START":
            self.counts[person_id]["arm"] += 1
            self.active[person_id]["arm"] = frame_sec

        elif arm_event == "END":
            start_ts = self.active[person_id]["arm"]
            if start_ts is not None:
                self.timestamps[person_id]["arm"].append({
                    "start": start_ts,
                    "end": frame_sec
                })
            self.active[person_id]["arm"] = None

        # -------------------------
        # LEG
        # -------------------------
        leg_event = self.leg.update(person_id, keypoints)

        if leg_event == "START":
            self.counts[person_id]["leg"] += 1
            self.active[person_id]["leg"] = frame_sec

        elif leg_event == "END":
            start_ts = self.active[person_id]["leg"]
            if start_ts is not None:
                self.timestamps[person_id]["leg"].append({
                    "start": start_ts,
                    "end": frame_sec
                })
            self.active[person_id]["leg"] = None

    # --------------------------------------------------
    def finalize(self, end_frame_sec):
        """
        Call once at end of analysis to close any open movements
        """
        for person_id, parts in self.active.items():
            for part, start_ts in parts.items():
                if start_ts is not None:
                    self.timestamps[person_id][part].append({
                        "start": start_ts,
                        "end": end_frame_sec
                    })
                    self.active[person_id][part] = None

    # --------------------------------------------------
    def mark_discontinued(self, person_id):
        self.counts[person_id]["discontinued"] = True

    def add_discontinuity(self, person_id, start, end):
        # Prevent duplicate entries
        if person_id in self.discontinued_once:
            return

        self.timestamps[person_id]["discontinuity"].append({
            "start": start,
            "end": end
        })

        self.discontinued_once.add(person_id)


    def get_discontinuities(self):
        return dict(self.timestamps)

    def register_person(self, person_id):
        _ = self.counts[person_id]      # force entry
        _ = self.timestamps[person_id]


    def get_counts(self, person_id):
        return self.counts[person_id]

    def get_all_counts(self):
        return dict(self.counts)

    def get_timestamps(self):
        return dict(self.timestamps)

