from dataclasses import dataclass
from typing import Dict, List, Optional
import cv2
import numpy as np
from actions.base import ActionDetector


@dataclass
class TrackState:
    prev_cx: float
    prev_cy: float
    moving_frames: int = 0
    idle_frames: int = 0
    state: str = "OCIOSO"  # default


class DefaultStateDetector(ActionDetector):
    """
    Regra simples:
      - Se pessoa existe:
          se deslocamento do centro do bbox (normalizado) > thr por N frames => SE_MOVENDO
          senão por M frames => OCIOSO
    """

    def __init__(
        self,
        thr: float = 0.010,        # 1% da diagonal do bbox por frame (ajuste)
        move_min_frames: int = 4,  # quantos frames seguidos p/ virar SE_MOVENDO
        idle_min_frames: int = 8,  # quantos frames seguidos p/ virar OCIOSO
        use_best_person_only: bool = True,
    ):
        self.thr = float(thr)
        self.move_min_frames = int(move_min_frames)
        self.idle_min_frames = int(idle_min_frames)
        self.use_best_person_only = use_best_person_only

        self.track: Optional[TrackState] = None
        self.events: List[dict] = []
        self.active_start: Optional[int] = None
        self.active_state: Optional[str] = None

    def _get_person_bbox(self, context: Dict):
        """
        Espera no context:
          - persons: lista de dicts com bbox [x1,y1,x2,y2] e opcional conf
        Se não tiver isso, você precisa preencher no seu yolo.detect().
        """
        persons = context.get("persons", []) or []
        if not persons:
            return None

        if self.use_best_person_only:
            persons = sorted(persons, key=lambda p: float(p.get("conf", 0.0)), reverse=True)
            return persons[0].get("bbox", None)

        return persons[0].get("bbox", None)

    def update(self, frame_idx: int, fps: float, frame, context: Dict) -> None:
        has_person = bool(context.get("has_person", False))
        if not has_person:
            context["default_state"] = None
            self._close_event(frame_idx, fps)
            self.track = None
            return

        bbox = self._get_person_bbox(context)
        if bbox is None:
            context["default_state"] = None
            return

        x1, y1, x2, y2 = map(float, bbox)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        diag = float(np.hypot((x2 - x1), (y2 - y1)) + 1e-6)

        if self.track is None:
            self.track = TrackState(prev_cx=cx, prev_cy=cy)
            context["default_state"] = self.track.state
            self._open_or_switch(frame_idx, fps, self.track.state)
            return

        dx = cx - self.track.prev_cx
        dy = cy - self.track.prev_cy
        disp = float(np.hypot(dx, dy) / diag)  # deslocamento normalizado

        # atualiza prev
        self.track.prev_cx, self.track.prev_cy = cx, cy

        if disp > self.thr:
            self.track.moving_frames += 1
            self.track.idle_frames = 0
        else:
            self.track.idle_frames += 1
            self.track.moving_frames = 0

        # histerese
        if self.track.moving_frames >= self.move_min_frames:
            self.track.state = "SE_MOVENDO"
        elif self.track.idle_frames >= self.idle_min_frames:
            self.track.state = "OCIOSO"

        context["default_state"] = self.track.state
        self._open_or_switch(frame_idx, fps, self.track.state)

    def _open_or_switch(self, frame_idx: int, fps: float, state: str):
        if self.active_state is None:
            self.active_state = state
            self.active_start = frame_idx
            return

        if state != self.active_state:
            # fecha anterior
            self._close_event(frame_idx, fps)
            # abre novo
            self.active_state = state
            self.active_start = frame_idx

    def _close_event(self, frame_idx: int, fps: float):
        if self.active_state is None or self.active_start is None:
            return
        start = int(self.active_start)
        end = int(frame_idx)
        dur = (end - start) / float(fps)
        self.events.append({
            "action": "default",
            "status": self.active_state,
            "event_id": f"df_{len(self.events):05d}",
            "start_frame": start,
            "end_frame": end,
            "duration_sec": round(float(dur), 2),
            "fps": float(fps),
        })
        self.active_state = None
        self.active_start = None

    def draw(self, frame, context: Dict) -> None:
        # não desenhar score, só ação no canto
        state = context.get("default_state", None)
        if not state:
            return
        cv2.putText(
            frame,
            f"SITUAÇÃO: {state}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    def flush(self, frame_idx: int, fps: float) -> List[dict]:
        self._close_event(frame_idx, fps)
        return self.events
