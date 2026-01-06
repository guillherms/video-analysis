import cv2
import mediapipe as mp


class MediaPipeHands:
    def __init__(self, max_hands=2, det_conf=0.5, track_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if not res.multi_hand_landmarks:
            return []

        out = []
        for lm in res.multi_hand_landmarks:
            # WRIST = landmark[0]
            wrist = lm.landmark[0]
            wx = int(wrist.x * w)
            wy = int(wrist.y * h)

            out.append({
                "wrist": (wx, wy),
            })

        return out

    def close(self):
        self.hands.close()
