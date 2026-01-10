import json
from pathlib import Path
from pipeline.actions_pipeline import run_detectors
from core.yolo_pose import YoloPoseDetector
from core.video_io import open_video, open_writer
from core.yolo_detector import YoloCocoDetector
from actions.using_phone import UsingPhoneDetector
from actions.meeting import MeetingDetector
from actions.default_state import DefaultStateDetector


def main():
    video_in = "data/input/video/video_analysis.mp4"

    video_out = "data/output/videos/events.mp4"
    jsonl_out = "data/output/videos/events.jsonl"

    Path("data/output/videos").mkdir(parents=True, exist_ok=True)

    cap, fps, w, h = open_video(video_in)
    out = open_writer(video_out, fps, w, h)

    yolo = YoloCocoDetector(model_path="yolo11n.pt", conf=0.35)
    pose = YoloPoseDetector(model_path="yolo11n-pose.pt", conf=0.25)

    detectors = [
        UsingPhoneDetector(),
        MeetingDetector(min_people=4),
        DefaultStateDetector(),
    ]

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = yolo.detect(frame)

        pose_dets = pose.detect(frame)
        pose_persons = []
        for pp in pose_dets.persons:
            pose_persons.append({
                "track_id": pp.track_id,
                "bbox": pp.bbox,
                "keypoints": pp.keypoints,
            })

        context = {
            "has_person": dets.has_person,
            "has_phone": dets.has_phone,
            "num_person": dets.num_person,
            "has_laptop": dets.has_laptop,
            "has_chair": dets.has_chair,
            "persons": dets.persons,
            "pose_persons": pose_persons,
        }

        run_detectors(
            detectors=detectors,
            frame=frame,
            frame_idx=frame_idx,
            fps=fps,
            context=context
        )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    all_events = []
    for det in detectors:
        all_events.extend(det.flush(frame_idx, fps))

    with open(jsonl_out, "w", encoding="utf-8") as f:
        for ev in all_events:
            f.write(json.dumps(ev) + "\n")

    print("✅ VIDEO:", video_out)
    print("✅ JSONL:", jsonl_out, "| events:", len(all_events))


if __name__ == "__main__":
    main()
