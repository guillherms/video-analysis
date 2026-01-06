from dataclasses import dataclass
from typing import Dict, List

import cv2
from actions.base import ActionDetector


@dataclass
class ActiveEvent:
    start_frame: int


class MeetingDetector(ActionDetector):
    """
    Regra simples (frame-level):
      MEETING = (num_person >= min_people) AND (has_laptop OR has_chair)
    """

    def __init__(self, min_people: int = 2):
        self.min_people = min_people
        self.active: ActiveEvent | None = None
        self.events: List[dict] = []

    def update(self, frame_idx: int, fps: float, frame, context: Dict) -> None:
        # context esperado: dict (global do frame)
        num_person = int(context.get("num_person", 0))
        has_laptop = bool(context.get("has_laptop", False))
        has_chair = bool(context.get("has_chair", False))

        meeting = (num_person >= self.min_people) and (has_laptop or has_chair)
        context["meeting"] = meeting

        if meeting and self.active is None:
            self.active = ActiveEvent(start_frame=frame_idx)

        if (not meeting) and self.active is not None:
            start = self.active.start_frame
            end = frame_idx
            dur = (end - start) / fps

            self.events.append({
                "action": "meeting",
                "event_id": f"mt_{len(self.events):05d}",
                "start_frame": int(start),
                "end_frame": int(end),
                "duration_sec": round(float(dur), 2),
                "fps": float(fps),
                "min_people": int(self.min_people),
            })
            self.active = None

    def draw(self, frame, context: Dict) -> None:
        if context.get("meeting", False):
            txt = f"AÇÃO: EM REUNIÃO/TRABALHANDO QTD. PESSOAS ={context.get('num_person', 0)}"
            cv2.putText(
                frame,
                txt,
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
                "action": "meeting",
                "event_id": f"mt_{len(self.events):05d}",
                "start_frame": int(start),
                "end_frame": int(end),
                "duration_sec": round(float(dur), 2),
                "fps": float(fps),
                "min_people": int(self.min_people),
            })
            self.active = None

        return self.events
