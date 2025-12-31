from reporting.schemas import empty_person_report
from reporting.rules import DEFAULT_RULES


class ReportBuilder:
    def __init__(self, rules=None):
        self.rules = rules if rules else DEFAULT_RULES

    def build(self, movement_counts, movement_timestamps):
        """
        movement_counts:
        {
            "person_1": {"neck": 2, "arm": 1, "leg": 0},
            "person_2": {"neck": 0, "arm": 0, "leg": 0}
        }

        movement_timestamps:
        {
            "person_1": {
                "neck": [{"start": "00:12:34", "end": "00:12:37"}],
                "discontinuity": [{"start": "01:15:20"}]
            }
        }
        """

        final_report = {}

        for person_id, counts in movement_counts.items():
            person_report = empty_person_report()
            timestamps = movement_timestamps.get(person_id, {})

            person_fail = False

            # -------------------------
            # PARTICIPANT DISCONTINUITY (REMARK ONLY)
            # -------------------------
            discontinuities = timestamps.get("discontinuity", [])

            if discontinuities:
                person_fail = True

                for d in discontinuities:
                    if isinstance(d, dict) and "start" in d:
                        person_report["remarks"].append(
                            f"Participant discontinued at {d['start']}"
                        )
                    else:
                        person_report["remarks"].append(
                            "Participant discontinued during session"
                        )

            # -------------------------
            # MOVEMENT EVALUATION
            # -------------------------
            for part in ["neck", "arm", "leg"]:
                allowed = self.rules[part]["max_allowed"]
                actual = counts.get(part, 0)

                person_report[part]["count"] = actual
                person_report[part]["allowed"] = allowed

                # Safe timestamp handling
                raw_ts = timestamps.get(part, [])
                if isinstance(raw_ts, list):
                    raw_ts = sorted(
                        raw_ts,
                        key=lambda x: x.get("start", "00:00:00")
                    )
                else:
                    raw_ts = []

                person_report[part]["timestamps"] = raw_ts

                if actual > allowed:
                    person_report[part]["status"] = "FAIL"
                    person_report["remarks"].append(
                        f"{part} movement exceeded limit ({actual} > {allowed})"
                    )
                    person_fail = True
                else:
                    person_report[part]["status"] = "PASS"

            

            # -------------------------
            # FINAL STATUS
            # -------------------------
            person_report["overall_status"] = "DISQUALIFIED" if person_fail else "PASSED"
            final_report[person_id] = person_report

        return final_report
