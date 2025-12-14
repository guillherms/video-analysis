import math

class ActionClassifier:
    """
    Classifica ações simples de cada pessoa com base nos dados:
    - posição da mão (MediaPipe Hands)
    - posição do celular (YOLO)
    """

    def __init__(self, phone_distance_threshold=80):
        self.phone_distance_threshold = phone_distance_threshold

    def classify(self, tracked_people, phone_boxes, hand_points):
        """
        tracked_people: list [(pid, (x1,y1,x2,y2)), ...]
        phone_boxes: list [(x1,y1,x2,y2), ...]
        hand_points: dict { pid: (hx,hy) or None }

        return: dict { pid: ["acao1", "acao2", ...] }
        """

        # Associar celulares à pessoa mais próxima
        phone_owner = self._assign_phones_to_people(tracked_people, phone_boxes)

        # Classificar ações
        actions = {}

        for pid, bbox in tracked_people:
            pid_actions = []

            # --------------- Mexendo no celular ---------------
            if self._is_using_phone(pid, phone_owner, hand_points):
                pid_actions.append("mexendo_celular")

            # ---------------------------------------------------
            # Futuras ações podem ser adicionadas aqui:
            # if self._is_reading(pid, ...):
            #     pid_actions.append("lendo_papel")
            #
            # if self._is_waving_hand(pid, ...):
            #     pid_actions.append("aceno_mao")
            # ---------------------------------------------------

            actions[pid] = pid_actions or ["nenhuma_acao"]

        return actions

    # ======================================================
    # DETECTAR "MEXENDO NO CELULAR"
    # ======================================================
    def _is_using_phone(self, pid, phone_owner, hand_points):
        """
        Determina se a mão da pessoa está próxima ao celular.
        """
        if pid not in phone_owner:
            return False

        if pid not in hand_points or hand_points[pid] is None:
            return False

        hx, hy = hand_points[pid]

        # pessoa pode estar segurando mais de 1 cel (raro, mas tratamos)
        for (px1, py1, px2, py2) in phone_owner[pid]:
            pcx = (px1 + px2) // 2
            pcy = (py1 + py2) // 2

            dist = math.hypot(hx - pcx, hy - pcy)

            if dist < self.phone_distance_threshold:
                return True

        return False

    # ======================================================
    # ASSOCIAR CELULARES À PESSOA MAIS PRÓXIMA
    # ======================================================
    def _assign_phones_to_people(self, tracked_people, phone_boxes):
        """
        phone_boxes -> lista de celulares detectados pelo YOLO
        Retorna dict: pid -> [lista de celulares]
        """
        phone_owner = {}

        for (px1, py1, px2, py2) in phone_boxes:
            pcx = (px1 + px2) // 2
            pcy = (py1 + py2) // 2

            best_pid = None
            best_dist = 999999

            for pid, (x1, y1, x2, y2) in tracked_people:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                dist = math.hypot(pcx - cx, pcy - cy)

                if dist < best_dist:
                    best_dist = dist
                    best_pid = pid

            # só atribui se o celular estiver razoavelmente perto
            if best_dist < 200:
                phone_owner.setdefault(best_pid, []).append((px1, py1, px2, py2))

        return phone_owner