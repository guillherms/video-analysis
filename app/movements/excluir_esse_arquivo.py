import cv2
from ultralytics import YOLO


def main():
    video_in = "data/input/video/video_analysis.mp4"
    video_out = "data/output/videos/yolo_mediapipe_observation.mp4"

    model = YOLO("C:\\Users\\guisa\\OneDrive\\Documentos\\Pos IADevs\\tech-challenge-4\\video-analysis\\yolo11n.pt")

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir o vídeo")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        video_out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    conf = 0.35

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=conf, verbose=False)[0]

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c, cls_id in zip(boxes, confs, clss):
                label = f"{model.names[cls_id]} {c:.2f}"

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (int(x1), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        out.write(frame)

    cap.release()
    out.release()

    print("✅ Vídeo gerado em:", video_out)


if __name__ == "__main__":
    main()