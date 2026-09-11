"""
detector.py
-----------
Thin wrapper around Ultralytics YOLO for real-time detection + tracking.
Keeps the rest of the app agnostic of the underlying model implementation,
so swapping YOLOv8 for YOLOv9/RT-DETR later only touches this file.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """A single tracked detection in one frame."""
    track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centroid(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1


class ObjectTracker:
    """
    Loads a YOLO model once and exposes a simple `track(frame)` call
    that returns a clean list of `Detection` objects using Ultralytics'
    built-in ByteTrack tracker (persistent IDs across frames).
    """

    # Ultralytics pretrained checkpoint names (downloaded on first use)
    AVAILABLE_MODELS = {
        "YOLOv8 Nano (fastest)": "yolov8n.pt",
        "YOLOv8 Small (balanced)": "yolov8s.pt",
        "YOLOv8 Medium (accurate)": "yolov8m.pt",
    }

    def __init__(self, model_key: str = "YOLOv8 Nano (fastest)"):
        weights = self.AVAILABLE_MODELS.get(model_key, "yolov8n.pt")
        self.model = YOLO(weights)
        self.class_names = self.model.names

    def reload(self, model_key: str):
        weights = self.AVAILABLE_MODELS.get(model_key, "yolov8n.pt")
        self.model = YOLO(weights)
        self.class_names = self.model.names

    def track(
        self,
        frame: np.ndarray,
        conf: float = 0.35,
        iou: float = 0.5,
        classes: Optional[List[int]] = None,
        tracker: str = "bytetrack.yaml",
    ) -> List[Detection]:
        """
        Run detection + tracking on a single BGR frame.
        Returns a list of Detection objects (empty list if nothing found).
        """
        results = self.model.track(
            frame,
            conf=conf,
            iou=iou,
            classes=classes,
            persist=True,
            tracker=tracker,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
        cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
        track_ids = (
            boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        )

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            cid = int(cls_ids[i])
            detections.append(
                Detection(
                    track_id=int(track_ids[i]) if track_ids[i] is not None else None,
                    class_id=cid,
                    class_name=self.class_names.get(cid, str(cid)),
                    confidence=float(confs[i]),
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                )
            )
        return detections
