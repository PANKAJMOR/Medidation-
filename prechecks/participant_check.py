from prechecks.base import BasePrecheck, PrecheckResult
from ultralytics import YOLO
import cv2

class ParticipantCheck(BasePrecheck):

    def __init__(self, min_people=1):
        self.model = YOLO("yolov8n.pt")
        self.min_people = min_people

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return PrecheckResult(False, "FRAME_READ_FAIL", "Cannot read frame")

        results = self.model(frame)[0]
        people = sum(int(box.cls[0] == 0) for box in results.boxes)

        if people < self.min_people:
            return PrecheckResult(
                False,
                "NO_PARTICIPANT",
                "No participant detected in video"
            )

        return PrecheckResult(True)
