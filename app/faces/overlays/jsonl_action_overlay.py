import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


@dataclass(frozen=True)
class ActionEvent:
    action: str                 # ex: "using_phone" ou "default"
    status: Optional[str]       # ex: "se movendo" (quando action == "default")
    event_id: str
    start_frame: int
    end_frame: int


def load_action_events(jsonl_path: str) -> List[ActionEvent]:
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL não encontrado: {jsonl_path}")

    events: List[ActionEvent] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            action = str(obj.get("action", "")).strip()
            status = str(obj.get("status", "")).strip() or None

            events.append(
                ActionEvent(
                    action=action,
                    status=status,
                    event_id=str(obj.get("event_id", "")).strip(),
                    start_frame=int(obj.get("start_frame", 0)),
                    end_frame=int(obj.get("end_frame", 0)),
                )
            )

    # garante ordem temporal (ajuda quando tem overlaps)
    events.sort(key=lambda e: (e.start_frame, e.end_frame, e.event_id))
    return events


def get_action_and_status_for_frame(
    events: List[ActionEvent],
    frame_idx: int,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Regra:
      - action: pega o evento ativo com action != "default" mais recente (maior start_frame)
      - status: pega o evento ativo com action == "default" mais recente (maior start_frame)

    Isso permite ter os dois ao mesmo tempo:
      {"action":"using_phone", ...}
      {"action":"default","status":"se movendo", ...}
    """
    action_val: Optional[str] = None
    status_val: Optional[str] = None
    best_action_start = -1
    best_status_start = -1

    for e in events:
        if not (e.start_frame <= frame_idx <= e.end_frame):
            continue

        if e.action and e.action.lower() != "default":
            if e.start_frame >= best_action_start:
                best_action_start = e.start_frame
                action_val = e.action

        if e.action and e.action.lower() == "default" and e.status:
            if e.start_frame >= best_status_start:
                best_status_start = e.start_frame
                status_val = e.status

    return action_val, status_val


def draw_action_and_status_top_left(
    frame,
    action_val: Optional[str],
    status_val: Optional[str],
) -> None:
    if not action_val and not status_val:
        return

    lines = []
    if action_val:
        lines.append(f"Action: {action_val}")
    if status_val:
        lines.append(f"Status: {status_val}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    color = (0, 255, 0)  # verde

    x = 20
    y0 = 40
    line_gap = 34

    for i, text in enumerate(lines):
        y = y0 + i * line_gap
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def open_video(video_path: str) -> Tuple[cv2.VideoCapture, float, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, w, h


def open_writer(video_path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Não consegui abrir writer para: {video_path}")
    return out


def overlay_jsonl_actions_on_video(
    video_in: str,
    jsonl_in: str,
    video_out: str,
) -> None:
    events = load_action_events(jsonl_in)

    cap, fps, w, h = open_video(video_in)
    out = open_writer(video_out, fps, w, h)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        action_val, status_val = get_action_and_status_for_frame(events, frame_idx)
        draw_action_and_status_top_left(frame, action_val, status_val)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print("✅ OVERLAY OK")
    print("✅ INPUT VIDEO:", video_in)
    print("✅ INPUT JSONL:", jsonl_in)
    print("✅ OUTPUT VIDEO:", video_out)
    print("✅ FRAMES:", frame_idx)


def main():
    # Exemplo comum no seu fluxo:
    # 1) seu script atual gera: data/output/videos/faces.mp4
    # 2) o pipeline de ações gera: data/output/videos/events.jsonl
    # 3) esse script gera o final: data/output/videos/faces_with_events.mp4
    video_in = "data/output/videos/faces.mp4"
    jsonl_in = "data/output/videos/events.jsonl"
    video_out = "data/output/videos/faces_with_events.mp4"

    overlay_jsonl_actions_on_video(
        video_in=video_in,
        jsonl_in=jsonl_in,
        video_out=video_out,
    )


if __name__ == "__main__":
    main()
