from ultralytics import YOLO


class PoseDetection:
    def __init__(self, bbox, keypoints, score):
        self.bbox = bbox          # [x1, y1, x2, y2]
        self.keypoints = keypoints
        self.score = score


class YOLOPoseDetector:
    def __init__(
        self,
        weights="yolov11n-pose.pt",
        conf=0.4,
        iou=0.5,
        imgsz=640        # 🔑 BEST SIZE FOR POSE
    ):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def detect(self, frame):
        """
        Runs YOLOv8-Pose with internal letterbox resizing.
        DO NOT resize frame before calling this.
        """

        results = self.model(
            frame,
            imgsz=self.imgsz,   # ✅ controlled resize
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )[0]

        detections = []

        if results.keypoints is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        keypoints = results.keypoints.xy.cpu().numpy()

        for box, score, kpts in zip(boxes, scores, keypoints):
            x1, y1, x2, y2 = box.astype(int)

            detections.append(
                PoseDetection(
                    bbox=[x1, y1, x2, y2],
                    keypoints=kpts,
                    score=float(score)
                )
            )

        return detections
