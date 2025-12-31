import itertools

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0


class IOUTracker:
    def __init__(self, iou_thresh=0.3):
        self.iou_thresh = iou_thresh
        self.tracks = {}        # id -> bbox
        self.next_id = 1

    def update(self, detections):
        """
        detections: list of (x1,y1,x2,y2)
        returns: list of (track_id, bbox)
        """
        updated_tracks = {}
        results = []

        used_ids = set()

        for det in detections:
            best_iou = 0
            best_id = None

            for tid, prev_box in self.tracks.items():
                if tid in used_ids:
                    continue
                score = iou(det, prev_box)
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_iou > self.iou_thresh:
                updated_tracks[best_id] = det
                results.append((best_id, det))
                used_ids.add(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                updated_tracks[tid] = det
                results.append((tid, det))
                used_ids.add(tid)

        self.tracks = updated_tracks
        return results
