from abc import ABC, abstractmethod
from training.framework.context import PluginContext
from training.framework.metadata import HealthReport

class BasePlugin(ABC):
    def __init__(self, context: PluginContext):
        self.context = context
        
    @abstractmethod
    def initialize(self) -> None:
        pass
        
    @abstractmethod
    def configure(self) -> None:
        pass
        
    @abstractmethod
    def start(self) -> None:
        pass
        
    @abstractmethod
    def stop(self) -> None:
        pass
        
    @abstractmethod
    def shutdown(self) -> None:
        pass
        
    @abstractmethod
    def health(self) -> HealthReport:
        pass
