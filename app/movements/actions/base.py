from abc import ABC, abstractmethod
from typing import Dict, List


class ActionDetector(ABC):
    @abstractmethod
    def update(self, frame_idx: int, fps: float, frame, context: Dict) -> None:
        ...

    @abstractmethod
    def draw(self, frame, context: Dict) -> None:
        ...

    @abstractmethod
    def flush(self, frame_idx: int, fps: float) -> List[dict]:
        ...
