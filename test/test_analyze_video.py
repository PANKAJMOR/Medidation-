# import sys
# import os
# import cv2
# import time

# # --------------------------------------------------
# # Add project root
# # --------------------------------------------------
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import json
# from pipeline.analyze_video import analyze_video

# VIDEO_PATH = r"D:\Meditation proctor\data\e87142b8\video.mp4.mkv"

# print("▶ Running analyze_video test...\n")

# result = analyze_video(VIDEO_PATH)

# print("▶ Raw Output:")
# print(json.dumps(result, indent=2))

# # -----------------------------
# # Basic assertions (manual)
# # -----------------------------
# if result["status"] == "FAILED":
#     print("\n❌ Analysis failed as expected")
#     for err in result.get("errors", []):
#         print(f"- {err['code']}: {err['message']}")

# else:
#     print("\n✅ Analysis completed successfully")

#     participants = result.get("participants", {})
#     role_map = result.get("role_mapping", {})
#     index_map = result.get("index_mapping", {})

#     print(f"Participants detected: {len(participants)}\n")

#     for person_id, report in participants.items():
#         print(
#             f"PersonID={person_id} | "
#             f"Index={index_map.get(person_id)} | "
#             f"Role={role_map.get(person_id)} | "
#             f"Overall={report['overall_status']}"
#         )

#         for part in ["neck", "arm", "leg"]:
#             p = report[part]
#             print(
#                 f"  - {part}: "
#                 f"{p['count']} / {p['allowed']} → {p['status']}"
#             )

#         if report["remarks"]:
#             print("  Remarks:")
#             for r in report["remarks"]:
#                 print(f"   • {r}")

#         print("-" * 40)

import sys
import os
import json
from datetime import timedelta

# --------------------------------------------------
# Add project root
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.analyze_video import analyze_video
from audio.audio_marker import AudioMarker

VIDEO_PATH = r"D:\Meditation proctor\data\bfcca4dc\video.mp4.mkv"

def sec_to_hhmmss(sec):
    return str(timedelta(seconds=int(sec)))

print("▶ Running analyze_video test...\n")

# --------------------------------------------------
# 1. AUDIO WINDOW (TEST VISIBILITY)
# --------------------------------------------------
print("▶ Extracting audio analysis window (test only)...")

audio_marker = AudioMarker()

try:
    start_sec, end_sec = audio_marker.get_analysis_window(
        VIDEO_PATH,
        start_ref=r"D:\Meditation proctor\reference_audio\start_audio.wav",
        end_ref=r"D:\Meditation proctor\reference_audio\end_audio.wav"
    )

    print(f"  Analysis START : {sec_to_hhmmss(start_sec)}")
    print(f"  Analysis END   : {sec_to_hhmmss(end_sec)}")
    print(f"  Duration       : {sec_to_hhmmss(end_sec - start_sec)}\n")

except Exception as e:
    print("❌ Audio marker failed:", e)
    print("Aborting test.")
    sys.exit(1)

# --------------------------------------------------
# 2. RUN FULL ANALYSIS
# --------------------------------------------------
result = analyze_video(VIDEO_PATH)

print("\n▶ Raw Output:")
print(json.dumps(result, indent=2))

# --------------------------------------------------
# 3. HUMAN-READABLE SUMMARY
# --------------------------------------------------
if result["status"] == "FAILED":
    print("\n❌ Analysis failed")
    for err in result.get("errors", []):
        print(f"- {err['code']}: {err['message']}")

else:
    print("\n✅ Analysis completed successfully")

    participants = result.get("participants", {})
    role_map = result.get("role_mapping", {})
    index_map = result.get("index_mapping", {})
    pdfs = result.get("pdf_reports", {})

    print(f"\nParticipants detected: {len(participants)}\n")

    for person_id, report in participants.items():
        print(
            f"PersonID={person_id} | "
            f"Index={index_map.get(person_id)} | "
            f"Role={role_map.get(person_id)} | "
            f"Overall={report['overall_status']}"
        )

        for part in ["neck", "arm", "leg"]:
            p = report[part]
            print(
                f"  - {part}: "
                f"{p['count']} / {p['allowed']} → {p['status']}"
            )

        if report["remarks"]:
            print("  Remarks:")
            for r in report["remarks"]:
                print(f"   • {r}")

        if person_id in pdfs:
            print(f"  📄 PDF: {pdfs[person_id]}")

        print("-" * 50)

print("\n✔ analyze_video test completed")
