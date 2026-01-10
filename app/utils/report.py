import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -------------------------
# Leitura helpers
# -------------------------
@dataclass(frozen=True)
class ActionEvent:
    action: str
    status: Optional[str]
    start_frame: int
    end_frame: int
    fps: float


@dataclass(frozen=True)
class EmotionSample:
    frame: int
    track_id: int
    emotion: str
    score: float
    fps: float


def read_jsonl(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -------------------------
# AÇÕES: contagem + duração
# -------------------------
def parse_action_events(events_jsonl: str) -> List[ActionEvent]:
    rows = read_jsonl(events_jsonl)
    out: List[ActionEvent] = []
    for r in rows:
        out.append(
            ActionEvent(
                action=str(r.get("action", "")).strip(),
                status=(str(r.get("status", "")).strip() or None),
                start_frame=int(r.get("start_frame", 0)),
                end_frame=int(r.get("end_frame", 0)),
                fps=float(r.get("fps", 30.0)),
            )
        )
    return out


def action_stats(events: List[ActionEvent]) -> Dict:
    # separa "ação" (action != default) e "status" (action == default)
    action_dur = defaultdict(float)
    status_dur = defaultdict(float)

    for e in events:
        duration_frames = max(0, e.end_frame - e.start_frame + 1)
        duration_sec = duration_frames / (e.fps or 30.0)

        if e.action.lower() == "default" and e.status:
            status_dur[e.status] += duration_sec
        elif e.action:
            action_dur[e.action] += duration_sec

    def to_percent_map(d: Dict[str, float]) -> Dict[str, float]:
        total = sum(d.values()) or 1.0
        return {k: round(v / total * 100.0, 2) for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)}

    return {
        "actions_duration_sec": dict(sorted(action_dur.items(), key=lambda x: x[1], reverse=True)),
        "actions_percent": to_percent_map(action_dur),
        "status_duration_sec": dict(sorted(status_dur.items(), key=lambda x: x[1], reverse=True)),
        "status_percent": to_percent_map(status_dur),
    }


# -------------------------
# EMOÇÕES: amostras -> % e duração aproximada
# -------------------------
def parse_emotion_samples(emotions_jsonl: str) -> List[EmotionSample]:
    rows = read_jsonl(emotions_jsonl)
    out: List[EmotionSample] = []
    for r in rows:
        out.append(
            EmotionSample(
                frame=int(r.get("frame", 0)),
                track_id=int(r.get("track_id", -1)),
                emotion=str(r.get("emotion", "")).strip(),
                score=float(r.get("score", 0.0)),
                fps=float(r.get("fps", 30.0)),
            )
        )
    return out


def emotion_stats(samples: List[EmotionSample], emo_every: int) -> Dict:
    # Cada sample representa aproximadamente emo_every frames de “estado”
    # (porque você calcula emoção a cada emo_every frames).
    dur = defaultdict(float)
    count = Counter()

    for s in samples:
        if not s.emotion:
            continue
        count[s.emotion] += 1
        dur[s.emotion] += (emo_every / (s.fps or 30.0))

    total_count = sum(count.values()) or 1
    total_dur = sum(dur.values()) or 1.0

    percent_count = {k: round(v / total_count * 100.0, 2) for k, v in count.most_common()}
    percent_dur = {k: round(v / total_dur * 100.0, 2) for k, v in sorted(dur.items(), key=lambda x: x[1], reverse=True)}

    return {
        "emotions_count": dict(count.most_common()),
        "emotions_percent_by_count": percent_count,
        "emotions_duration_sec_approx": dict(sorted(dur.items(), key=lambda x: x[1], reverse=True)),
        "emotions_percent_by_duration_approx": percent_dur,
    }
    

def write_csv(
    path: str,
    headers: List[str],
    rows: List[Dict[str, object]],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -------------------------
# Report
# -------------------------
def main():
    events_jsonl = "data/output/videos/events.jsonl"
    emotions_jsonl = "data/output/videos/emotions.jsonl"  # você vai gerar esse
    report_out = "data/output/reports/report.json"

    emo_every = 10

    events = parse_action_events(events_jsonl)
    a_stats = action_stats(events)

    samples = parse_emotion_samples(emotions_jsonl)
    e_stats = emotion_stats(samples, emo_every=emo_every)

    report = {
        "actions": a_stats,
        "emotions": e_stats,
    }

    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # print top no console
    print("✅ REPORT:", report_out)

    print("\nTop AÇÕES (% por duração):")
    for k, v in list(report["actions"]["actions_percent"].items())[:5]:
        print(f"  - {k}: {v}%")

    print("\nTop SITUAÇÕES (% por duração):")
    for k, v in list(report["actions"]["status_percent"].items())[:5]:
        print(f"  - {k}: {v}%")

    print("\nTop EMOÇÕES (% por duração aprox):")
    for k, v in list(report["emotions"]["emotions_percent_by_duration_approx"].items())[:5]:
        print(f"  - {k}: {v}%")

    actions_csv = "data/output/reports/actions_summary.csv"
    status_csv = "data/output/reports/status_summary.csv"
    emotions_csv = "data/output/reports/emotions_summary.csv"

    # ações
    action_rows = []
    for action, dur in a_stats["actions_duration_sec"].items():
        action_rows.append(
            {
                "action": action,
                "duration_sec": round(dur, 2),
                "percent": a_stats["actions_percent"].get(action, 0.0),
            }
        )

    write_csv(
        actions_csv,
        headers=["action", "duration_sec", "percent"],
        rows=action_rows,
    )

    # status
    status_rows = []
    for status, dur in a_stats["status_duration_sec"].items():
        status_rows.append(
            {
                "status": status,
                "duration_sec": round(dur, 2),
                "percent": a_stats["status_percent"].get(status, 0.0),
            }
        )

    write_csv(
        status_csv,
        headers=["status", "duration_sec", "percent"],
        rows=status_rows,
    )

    # emoções
    emotion_rows = []
    for emo, dur in e_stats["emotions_duration_sec_approx"].items():
        emotion_rows.append(
            {
                "emotion": emo,
                "duration_sec_approx": round(dur, 2),
                "percent": e_stats["emotions_percent_by_duration_approx"].get(emo, 0.0),
            }
        )

    write_csv(
        emotions_csv,
        headers=["emotion", "duration_sec_approx", "percent"],
        rows=emotion_rows,
    )

    print("✅ CSVs gerados:")
    print(" -", actions_csv)
    print(" -", status_csv)
    print(" -", emotions_csv)


if __name__ == "__main__":
    main()
