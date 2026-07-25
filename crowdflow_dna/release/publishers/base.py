from abc import ABC, abstractmethod

class Publisher(ABC):
    @abstractmethod
    def publish(self, release_bundle: dict):
        pass
