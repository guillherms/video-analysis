import os
from moviepy.editor import VideoFileClip

video = VideoFileClip("data/input/video/video_analysis.mp4")

# Exemplo de cortes (em segundos)
clips = [
    (0, 5, "reading"),
    (17, 36, "nothing"),
]

os.makedirs("data/output/videos_cut_scene", exist_ok=True)

for start, end, label in clips:
    sub = video.subclip(start, end)
    output_path = f"data/output/videos_cut_scene/{label}.mp4"
    sub.write_videofile(output_path, codec="libx264", audio=False)

print("All scenes exported successfully!")