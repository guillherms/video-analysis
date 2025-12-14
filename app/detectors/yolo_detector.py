from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect(self, frame):
        """
        Retorna:
        - people: lista de bboxes [(x1,y1,x2,y2), ...]
        - phones: lista de bboxes [(x1,y1,x2,y2), ...]
        """
        results = self.model(frame, verbose=False)[0]

        people = []
        phones = []

        for box in results.boxes:
            cls = int(box.cls[0])
            label = results.names[cls]

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            if label == "person":
                people.append((x1, y1, x2, y2))
            elif label == "cell phone":
                phones.append((x1, y1, x2, y2))

        return people, phones
