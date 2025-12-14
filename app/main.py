from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")


@dataclass
class EmotionCacheItem:
    emotion: str
    score: float
    frame_idx: int


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def analyze_emotion_deepface(face_bgr: np.ndarray) -> Optional[Tuple[str, float]]:
    try:
        res = DeepFace.analyze(
            img_path=face_bgr,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
        )
        if isinstance(res, list):
            res = res[0]
        emo = res.get("dominant_emotion", "")
        emo_scores = res.get("emotion", {}) or {}
        score = float(emo_scores.get(emo, 0.0))
        if not emo:
            return None
        return emo, score
    except Exception:
        return None


def main():
    video_in = "data/input/video/facial_rcecognition_activities_analysis.mp4"
    video_out = "data/output/videos/testing2.mp4"

    MIN_DRAW_CONF = 0.65   # 👈 ajuste aqui (0.60 a 0.75)
    conf_det = 0.20
    emo_every = 10
    min_face = 40
    tracker = "botsort.yaml"

    TTL_FRAMES = 15
    last_bbox_by_id: Dict[int, Dict[str, object]] = {}

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir o vídeo: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

    model = YOLO(MODEL_PATH)

    emo_cache: Dict[int, EmotionCacheItem] = {}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            frame,
            persist=True,
            conf=conf_det,
            iou=0.5,
            tracker=tracker,
            verbose=False
        )
        r = results[0]

        drawn_ids = set()

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            ids = r.boxes.id
            ids = ids.cpu().numpy().astype(int) if ids is not None else None

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                det_conf = float(confs[i])

                # ✅ ÚNICA MUDANÇA: filtra deteccoes fracas (evita marcar mão/ruído)
                if det_conf < MIN_DRAW_CONF:
                    continue

                bbox = clamp_bbox(x1, y1, x2, y2, w, h)
                if bbox is None:
                    continue
                x1c, y1c, x2c, y2c = bbox

                bw, bh = (x2c - x1c), (y2c - y1c)
                if bw < min_face or bh < min_face:
                    continue

                track_id = int(ids[i]) if ids is not None else -1

                face_roi = frame[y1c:y2c, x1c:x2c]

                if track_id != -1:
                    last_bbox_by_id[track_id] = {
                        "bbox": (x1c, y1c, x2c, y2c),
                        "last_seen": frame_idx
                    }
                    drawn_ids.add(track_id)

                do_emo = (track_id != -1) and (
                    (track_id not in emo_cache) or
                    ((frame_idx - emo_cache[track_id].frame_idx) >= emo_every)
                )

                if do_emo and face_roi.size > 0:
                    emo_res = analyze_emotion_deepface(face_roi)
                    if emo_res is not None:
                        emo, score = emo_res
                        emo_cache[track_id] = EmotionCacheItem(emo, score, frame_idx)

                emo_text = ""
                if track_id in emo_cache:
                    ec = emo_cache[track_id]
                    emo_text = f"{ec.emotion} {ec.score:.1f}"

                label = f"id:{track_id} conf:{det_conf:.2f} {emo_text}".strip()

                cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1c, max(20, y1c - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # TTL fallback (mantém bbox por alguns frames)
        for track_id, data in list(last_bbox_by_id.items()):
            if track_id in drawn_ids:
                continue

            last_seen = int(data["last_seen"])
            if frame_idx - last_seen > TTL_FRAMES:
                del last_bbox_by_id[track_id]
                continue

            x1c, y1c, x2c, y2c = data["bbox"]

            emo_text = ""
            if track_id in emo_cache:
                ec = emo_cache[track_id]
                emo_text = f"{ec.emotion} {ec.score:.1f}"

            label = f"id:{track_id} {emo_text}".strip()

            cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0, 200, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1c, max(20, y1c - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
                cv2.LINE_AA
            )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("✅ OK:", video_out)


if __name__ == "__main__":
    main()
