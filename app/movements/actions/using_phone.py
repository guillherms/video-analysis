from dataclasses import dataclass
from typing import Dict, List

import cv2
from actions.base import ActionDetector


@dataclass
class ActiveEvent:
    start_frame: int


class UsingPhoneDetector(ActionDetector):
    """
    Regra simples:
    - Se tem pessoa e tem celular no mesmo frame -> USING_PHONE = True
    """

    def __init__(self):
        self.active: ActiveEvent | None = None
        self.events: List[dict] = []

    def update(self, frame_idx: int, fps: float, frame, context: Dict) -> None:
        if not isinstance(context, dict):
            raise TypeError(
                f"context precisa ser dict, recebido: {type(context)}"
            )
        has_person = context.get("has_person", False)
        has_phone = context.get("has_phone", False)

        using_phone = bool(has_person and has_phone)
        context["using_phone"] = using_phone

        if using_phone and self.active is None:
            self.active = ActiveEvent(start_frame=frame_idx)

        if (not using_phone) and self.active is not None:
            start = self.active.start_frame
            end = frame_idx
            dur = (end - start) / fps

            self.events.append({
                "action": "using_phone",
                "event_id": f"ph_{len(self.events):05d}",
                "start_frame": int(start),
                "end_frame": int(end),
                "duration_sec": round(float(dur), 2),
                "fps": float(fps),
            })
            self.active = None

    def draw(self, frame, context: Dict) -> None:
        if context.get("using_phone", False):
            cv2.putText(
                frame,
                "AÇÃO: USANDO CELULAR",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    def flush(self, frame_idx: int, fps: float) -> List[dict]:
        if self.active is not None:
            start = self.active.start_frame
            end = frame_idx - 1
            dur = (end - start) / fps
            self.events.append({
                "action": "using_phone",
                "event_id": f"ph_{len(self.events):05d}",
                "start_frame": int(start),
                "end_frame": int(end),
                "duration_sec": round(float(dur), 2),
                "fps": float(fps),
            })
            self.active = None

        return self.events
