from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from Models import SiameseNetwork  # your model definition

class MetricWrapper(ABC):
    @abstractmethod
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        pass

    def get_params(self):
        return {}
    
    def assert_non_negative(self, value: float) -> None:
        if value < 0:
            raise ValueError("Metric value must be non-negative.")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.get_params()})"
    
class WeightedEuclideanMetric(MetricWrapper):
    def __init__(self, weights: np.ndarray):
        self.weights = np.asarray(weights)

    def __call__(self, x1, x2):
        diff = x1 - x2
        value = np.sqrt(np.sum(self.weights * diff ** 2))
        self.assert_non_negative(value)
        return value

    def get_params(self):
        return {'weights': self.weights.tolist()}
    
class EuclideanMetric(MetricWrapper):
    def __call__(self, x1, x2):
        print("Testing")
        value = np.linalg.norm(x1 - x2)
        self.assert_non_negative(value)
        return value

    def get_params(self):
        return {}
    
class ConstantMetric(MetricWrapper):
    def __init__(self, value = 1.0):
        if value < 0:
            raise ValueError("Constant metric value must be non-negative.")
        self.value = value

    def __call__(self, x1, x2):
        self.assert_non_negative(self.value)
        return self.value

    def get_params(self):
        return {'value': self.value}
    
class InvertedEuclideanMetric(MetricWrapper):
    def __call__(self, x1, x2):
        value = 1.0 / np.linalg.norm(x1 - x2) if np.linalg.norm(x1 - x2) != 0 else float('inf')
        self.assert_non_negative(value)
        return value

    def get_params(self):
        return {}

class SiameseNetworkMetric(MetricWrapper):
    def __init__(
        self,
        model: SiameseNetwork,
        device: str = "cpu",
        threshold: Optional[float] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold 

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        t1 = torch.tensor(x1, dtype=torch.float32, device=self.device).unsqueeze(0)
        t2 = torch.tensor(x2, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            e1 = self.model.forward_once(t1)
            e2 = self.model.forward_once(t2)
            distance = F.pairwise_distance(e1, e2).item()

        self.assert_non_negative(distance)
        return distance

    def get_params(self):
        return {"model": "SiameseNetwork", "device": self.device}
