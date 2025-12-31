from utils.time_formatter import seconds_to_hhmmss


def convert_movement_timestamps(
    movement_timestamps: dict,
    base_offset_sec: float
):
    """
    Converts movement timestamps (relative to analysis window)
    into absolute video timeline timestamps.

    movement_timestamps:
    {
        "person_1": {
            "neck": [
                {"start": 120.4, "end": 123.1}
            ]
        }
    }
    """

    converted = {}

    for person_id, parts in movement_timestamps.items():
        converted[person_id] = {}

        for part, events in parts.items():
            converted[person_id][part] = []

            for e in events:
                start_abs = base_offset_sec + e["start"]
                end_abs   = base_offset_sec + e["end"]

                converted[person_id][part].append({
                    "start": seconds_to_hhmmss(start_abs),
                    "end": seconds_to_hhmmss(end_abs),
                    "duration_sec": round(end_abs - start_abs, 2)
                })

            # Handle discontinuity separately
            disc_events = parts.get("discontinuity", [])
            converted[person_id]["discontinuity"] = []

            for e in disc_events:
                start_rel = e["start"] + base_offset_sec
                converted[person_id]["discontinuity"].append({
                    "start": seconds_to_hhmmss(start_rel)
                })


    return converted
