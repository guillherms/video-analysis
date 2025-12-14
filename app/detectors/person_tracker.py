import numpy as np

class PersonTracker:
    def __init__(self, max_distance=80):
        self.next_id = 0
        self.centroids = {}   # id -> (cx, cy)
        self.max_distance = max_distance

    def assign_ids(self, detections):
        """
        detections = [(x1,y1,x2,y2), ...]
        return: [(id, (x1,y1,x2,y2)), ...]
        """
        assigned = []
        new_centroids = {}

        for bbox in detections:
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            best_id = None
            min_dist = 999999

            # tenta encontrar ID antigo próximo deste centro
            for pid, (px, py) in self.centroids.items():
                dist = np.hypot(cx - px, cy - py)
                if dist < min_dist and dist < self.max_distance:
                    best_id = pid
                    min_dist = dist

            # se ninguém encaixa → novo ID
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            new_centroids[best_id] = (cx, cy)
            assigned.append((best_id, bbox))

        self.centroids = new_centroids
        return assigned
