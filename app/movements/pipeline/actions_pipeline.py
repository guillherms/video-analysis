from typing import List, Dict
from actions.base import ActionDetector


def run_detectors(
    detectors: List[ActionDetector],
    frame,
    frame_idx: int,
    fps: float,
    context: Dict
):
    # update
    for det in detectors:
        det.update(frame_idx, fps, frame, context)

    # draw
    for det in detectors:
        det.draw(frame, context)
