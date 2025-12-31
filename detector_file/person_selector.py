# meditation_proctor/detector/person_selector.py

class PersonSelector:
    """
    Selects at most 3 closest people, filters out back-row people using height threshold,
    and assigns LEFT / CENTER / RIGHT roles.
    """

    def __init__(self, min_height=110, max_people=3):
        self.min_height = min_height
        self.max_people = max_people

    def select_people(self, detections):
        """
        detections: YOLO detections having .x1, .y1, .x2, .y2

        Returns:
            roles: dict → {"left": det, "center": det, "right": det}
            error: None or error string
        """

        # ----------------------------------------
        # Step 1 — Remove small bounding boxes (back row)
        # ----------------------------------------
        filtered = []
        for det in detections:
            h = det.y2 - det.y1
            if h >= self.min_height:   # keep only front-row people
                x_center = (det.x1 + det.x2) / 2
                y_center = (det.y1 + det.y2) / 2
                filtered.append((det, x_center, y_center))

        if len(filtered) == 0:
            return None, "No visible people detected. Check lighting/angle/video quality."

        # ----------------------------------------
        # Step 2 — Take closest people (largest y-center)
        # ----------------------------------------
        filtered.sort(key=lambda x: x[2], reverse=True)
        selected = filtered[: self.max_people]

        # ----------------------------------------
        # Step 3 — Sort by x-center (left → right)
        # ----------------------------------------
        selected.sort(key=lambda x: x[1])
        persons = [item[0] for item in selected]

        # ----------------------------------------
        # Step 4 — Assign roles based on count
        # ----------------------------------------
        roles = {}

        if len(persons) == 1:
            roles["center"] = persons[0]

        elif len(persons) == 2:
            roles["left"] = persons[0]
            roles["right"] = persons[1]

        elif len(persons) == 3:
            roles["left"]   = persons[0]
            roles["center"] = persons[1]
            roles["right"]  = persons[2]

        return roles, None
