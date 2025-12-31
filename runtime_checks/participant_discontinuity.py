from collections import defaultdict


class ParticipantDiscontinuity:
    def __init__(self, max_absent_seconds, fps):
        self.max_absent_seconds = max_absent_seconds
        self.fps = fps

        self.last_seen = {}
        self.active_absence = {}
        self.discontinuities = defaultdict(list)

    def update(self, person_id, current_sec):
        # Person visible now
        self.last_seen[person_id] = current_sec

        # If absence was active → END it
        if person_id in self.active_absence:
            start = self.active_absence.pop(person_id)
            self.discontinuities[person_id].append({
                "start": start,
                "end": current_sec
            })

    def check(self, current_sec):
        discontinued = []

        for pid, last in self.last_seen.items():
            if current_sec - last >= self.max_absent_seconds:
                if pid not in self.active_absence:
                    self.active_absence[pid] = last
                discontinued.append(pid)

        return discontinued

    def get_timestamps(self):
        return dict(self.discontinuities)

    def is_discontinued(self, person_id):
        return person_id in self.discontinued
