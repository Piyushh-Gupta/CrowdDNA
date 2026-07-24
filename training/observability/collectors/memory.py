try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from training.observability.registry import ObservabilityRegistry, ModuleMetadata, ObservabilityModule
from training.observability.metadata import MetricObservation
from training.observability.buffer import ObservationBuffer
from training.observability.clock import Clock

@ObservabilityRegistry.register(ModuleMetadata(name="memory_collector", version="1.0", schema_version="v1", category="collector"))
class MemoryCollector(ObservabilityModule):
    def __init__(self):
        self.buffer = None
        
    def initialize(self, buffer: ObservationBuffer = None) -> None:
        self.buffer = buffer
        
    def start(self) -> None:
        pass
        
    def stop(self) -> None:
        pass
        
    def shutdown(self) -> None:
        pass
        
    def collect(self) -> None:
        if self.buffer is None or not HAS_PSUTIL:
            return
            
        mem = psutil.virtual_memory()
        obs = MetricObservation(
            timestamp=Clock.now(),
            name="memory_used",
            value=mem.used / (1024 * 1024),
            unit="MB"
        )
        self.buffer.add(obs)
