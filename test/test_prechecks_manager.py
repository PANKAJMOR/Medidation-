import sys
import os

# --------------------------------------------------
# Add project root
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prechecks.precheck_manager import PrecheckManager
from prechecks.video_access import VideoAccessCheck
from prechecks.video_metadata import VideoMetadataCheck
from prechecks.illumination import IlluminationCheck
from prechecks.freeze_detection import FreezeCheck
from prechecks.participant_check import ParticipantCheck
from prechecks.timestamp_check import TimestampCheck

# --------------------------------------------------
# Config
# --------------------------------------------------
VIDEO_PATH = r"D:\Meditation proctor\data\ef81e229\video.mp4"

print("\n▶ Running Prechecks Manager Test (Client Error Mapping Enabled)\n")

# --------------------------------------------------
# Initialize checks
# --------------------------------------------------
precheck_manager = PrecheckManager([
    VideoAccessCheck(),
    VideoMetadataCheck(min_duration_sec=2.75 * 3600),   # 2h 45m
    IlluminationCheck(min_brightness=40),
    FreezeCheck(
        sample_frames=60,
        diff_thresh=1.0,
        identical_ratio_thresh=1.0
    ),
    ParticipantCheck(min_people=1),
    TimestampCheck(required_year=2026)   # can be extended to year check
])

# --------------------------------------------------
# Run prechecks
# --------------------------------------------------
result = precheck_manager.run_all(VIDEO_PATH)

# --------------------------------------------------
# Display results (CLIENT FORMAT)
# --------------------------------------------------
if result["passed"]:
    print("✅ PRECHECKS PASSED — Video is valid for analysis\n")
else:
    print("❌ PRECHECKS FAILED\n")
    for i, err in enumerate(result["errors"], start=1):
        print(f"{i}. {err['code']}")
        print(f"   {err['message']}\n")

print("✔ Prechecks manager test completed\n")
