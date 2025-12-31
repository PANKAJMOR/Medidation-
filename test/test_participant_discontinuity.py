import sys
import os

# --------------------------------------------------
# Add project root
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from runtime_checks.participant_discontinuity import ParticipantDiscontinuity

def run_test():
    FPS = 1  # 1 FPS (same as analyze_video)
    MAX_ABSENT_SECONDS = 10  # short for testing

    monitor = ParticipantDiscontinuity(
        max_absent_seconds=MAX_ABSENT_SECONDS,
        fps=FPS
    )

    print("## Starting Participant Discontinuity Test\n")

    # -----------------------------------------
    # Simulate frames
    # -----------------------------------------
    for frame in range(1, 31):

        # person_1 visible from frame 1–5
        if frame <= 5:
            monitor.update("person_1", frame)

        # person_1 disappears from frame 6 onwards
        # threshold = 10 sec → should discontinue at frame 16

        discontinued = monitor.check(frame)

        print(
            f"Frame {frame:02d} | "
            f"Discontinued: {discontinued}"
        )

    # -----------------------------------------
    # Assertions
    # -----------------------------------------
    assert monitor.is_discontinued("person_1"), \
        " person_1 should be discontinued"

    print("\n Test PASSED — Participant correctly marked discontinued")


if __name__ == "__main__":
    run_test()
