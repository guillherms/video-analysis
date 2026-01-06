from dataclasses import dataclass
from typing import List, Dict
from ultralytics import YOLO

PERSON_ID = 0
CELL_PHONE_ID = 67
LAPTOP_ID = 63
CHAIR_ID = 56


@dataclass
class FrameDetections:
    has_person: bool
    has_phone: bool
    num_person: int
    has_laptop: bool
    has_chair: bool
    persons: List[Dict]  # <- NOVO


class YoloCocoDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame) -> FrameDetections:
        r = self.model(frame, conf=self.conf, verbose=False)[0]

        num_person = 0
        has_phone = False
        has_laptop = False
        has_chair = False
        persons: List[Dict] = []

        if r.boxes is not None and len(r.boxes) > 0:
            clss = r.boxes.cls.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()          # (N,4)
            confs = r.boxes.conf.cpu().numpy()         # (N,)

            for i, c in enumerate(clss):
                if c == PERSON_ID:
                    num_person += 1
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    persons.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": float(confs[i]),
                    })
                elif c == CELL_PHONE_ID:
                    has_phone = True
                elif c == LAPTOP_ID:
                    has_laptop = True
                elif c == CHAIR_ID:
                    has_chair = True

        has_person = num_person > 0

        return FrameDetections(
            has_person=has_person,
            has_phone=has_phone,
            num_person=num_person,
            has_laptop=has_laptop,
            has_chair=has_chair,
            persons=persons,
        )
