from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class PosePerson:
    track_id: int
    bbox: Tuple[int, int, int, int]          # x1,y1,x2,y2
    keypoints: np.ndarray                   # (K, 3) -> x,y,conf


@dataclass
class PoseDetections:
    persons: List[PosePerson]


class YoloPoseDetector:
    """
    Usa YOLO Pose com tracker para ter track_id por pessoa.
    Modelos típicos:
      - yolov8n-pose.pt
      - yolo11n-pose.pt (se você já baixou)
    """
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.5, tracker: str = "botsort.yaml"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.tracker = tracker

    def detect(self, frame) -> PoseDetections:
        res = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker,
            verbose=False
        )[0]

        persons: List[PosePerson] = []

        if res.boxes is None or len(res.boxes) == 0:
            return PoseDetections(persons=[])

        # ids (track)
        ids = res.boxes.id
        if ids is None:
            # sem track id -> ignora (ou seta -1)
            return PoseDetections(persons=[])

        ids = ids.cpu().numpy().astype(int)
        boxes = res.boxes.xyxy.cpu().numpy().astype(float)

        # keypoints: (N, K, 3)
        if res.keypoints is None:
            return PoseDetections(persons=[])

        kpts = res.keypoints.data  # torch tensor
        kpts = kpts.cpu().numpy().astype(float)

        for i in range(len(ids)):
            x1, y1, x2, y2 = boxes[i]
            person = PosePerson(
                track_id=int(ids[i]),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                keypoints=kpts[i]  # (K,3)
            )
            persons.append(person)

        return PoseDetections(persons=persons)
