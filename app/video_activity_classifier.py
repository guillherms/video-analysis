import os
from moviepy.editor import VideoFileClip

video = VideoFileClip("data/input/video/facial_analysis.mp4")

# Exemplo de cortes (em segundos)
clips = [
    (0, 5, "reading"),
    (6, 8, "waving"),
    (10, 12, "agreeing"),
    (18, 22, "dancing"),
    (36, 41, "lying_down"),
    (61, 67, "medical_procedure"),
    (68, 71, "medical_observation"),
    (80, 90, "office_work"),
    (99, 102, "handshake"),
    (104, 108, "office_work"),
]

os.makedirs("data/output/videos", exist_ok=True)

for start, end, label in clips:
    sub = video.subclip(start, end)
    output_path = f"data/output/videos/{label}.mp4"
    sub.write_videofile(output_path, codec="libx264", audio=False)

print("All scenes exported successfully!")