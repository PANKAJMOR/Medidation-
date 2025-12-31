import os
import datetime
from prechecks.base import BasePrecheck, PrecheckResult

class TimestampCheck(BasePrecheck):

    def __init__(self, required_year=2026):
        self.required_year = required_year

    def run(self, video_path):
        try:
            modified = os.path.getmtime(video_path)
        except Exception:
            return PrecheckResult(
                False,
                "VIDEO_NOT_ACCESSIBLE",
                "Cannot read video timestamp"
            )

        year = datetime.datetime.fromtimestamp(modified).year

        if year != self.required_year:
            return PrecheckResult(
                False,
                "INCORRECT_VIDEO_TIMESTAMP",
                f"Video timestamp year {year} != {self.required_year}"
            )

        return PrecheckResult(True)
