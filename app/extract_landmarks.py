import os, mediapipe as mp, cv2, csv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
csv_path = "data/output/files/dataset_pose.csv"

header = [f"{a}{i}" for i in range(33) for a in ("x","y","z","v")] + ["label"]
if not os.path.exists(csv_path):
    open(csv_path,"w").write(",".join(header)+"\n")

for file in os.listdir("data/output/videos"):
    label = os.path.splitext(file)[0]
    cap = cv2.VideoCapture(os.path.join("data/output/videos", file))
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            row = [str(v) for lm in res.pose_landmarks.landmark for v in (lm.x,lm.y,lm.z,lm.visibility)]
            open(csv_path,"a").write(",".join(row+[label])+"\n")
    cap.release()
print("✅ Dataset salvo em", csv_path)