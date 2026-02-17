from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def train(self, samples):
        pass

    @abstractmethod
    def predict(self, code: str) -> float:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass