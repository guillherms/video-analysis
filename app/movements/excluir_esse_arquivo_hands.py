import cv2
from ultralytics import YOLO

VIDEO_IN = "data/output/videos_cut_scene/nothing.mp4"
VIDEO_OUT = "data/output/videos/yolo_test.mp4"
MODEL_PATH = "yolo11n-pose.pt"

def main():
    # load model
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir o vídeo")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        VIDEO_OUT,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # inference
        result = model(frame, conf=0.25, verbose=False)[0]

        # desenha automaticamente bbox + skeleton
        annotated = result.plot()

        # info simples no canto
        num_person = len(result.boxes) if result.boxes is not None else 0
        cv2.putText(
            annotated,
            f"Frame {frame_idx} | persons={num_person}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()

    print("✅ vídeo gerado:", VIDEO_OUT)


if __name__ == "__main__":
    main()