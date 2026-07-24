from training.observability.registry import ObservabilityRegistry, ModuleMetadata, ObservabilityModule
from training.observability.metadata import MetricObservation
from training.observability.buffer import ObservationBuffer
from training.observability.clock import Clock

@ObservabilityRegistry.register(ModuleMetadata(name="queue_collector", version="1.0", schema_version="v1", category="collector"))
class QueueCollector(ObservabilityModule):
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
        
    def record_queue_depth(self, name: str, depth: int) -> None:
        if self.buffer is None:
            return
            
        obs = MetricObservation(
            timestamp=Clock.now(),
            name=f"queue_{name}_depth",
            value=float(depth),
            unit="items"
        )
        self.buffer.add(obs)
