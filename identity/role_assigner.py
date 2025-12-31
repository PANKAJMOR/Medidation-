# identity/role_assigner.py

class RoleAssigner:
    def __init__(self):
        self.assigned = False
        self.role_map = {}    # person_id -> "left"/"center"/"right"
        self.index_map = {}   # person_id -> fixed index

    def assign(self, tracked_people):
        """
        tracked_people: list of (person_id, bbox)
        bbox = [x1, y1, x2, y2]
        """

        if self.assigned:
            return self.role_map, self.index_map

        # Sort by horizontal position (x-center)
        sorted_people = sorted(
            tracked_people,
            key=lambda p: (p[1][0] + p[1][2]) / 2
        )

        n = len(sorted_people)

        if n == 3:
            roles = ["left", "center", "right"]
        elif n == 2:
            roles = ["left", "right"]
        elif n == 1:
            roles = ["center"]
        else:
            raise ValueError("Unsupported number of participants")

        for idx, ((person_id, _), role) in enumerate(zip(sorted_people, roles)):
            self.role_map[person_id] = role
            self.index_map[person_id] = idx

        self.assigned = True
        return self.role_map, self.index_map
