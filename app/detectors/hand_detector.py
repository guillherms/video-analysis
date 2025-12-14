import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame, tracked_people):
        """
        tracked_people: [(pid, (x1,y1,x2,y2)), ...]
        Retorno: dict { pid: (hx,hy) ou None }
        """
        hand_points = {}

        for pid, (x1, y1, x2, y2) in tracked_people:
            person_img = frame[y1:y2, x1:x2]
            rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            result = self.hands.process(rgb)

            if result.multi_hand_landmarks:
                hlm = result.multi_hand_landmarks[0]
                wrist = hlm.landmark[0]

                hx = int(x1 + wrist.x * (x2 - x1))
                hy = int(y1 + wrist.y * (y2 - y1))

                hand_points[pid] = (hx, hy)
            else:
                hand_points[pid] = None

        return hand_points
