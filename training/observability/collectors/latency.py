from training.observability.registry import ObservabilityRegistry, ModuleMetadata, ObservabilityModule
from training.observability.metadata import MetricObservation
from training.observability.buffer import ObservationBuffer
from training.observability.clock import Clock

@ObservabilityRegistry.register(ModuleMetadata(name="latency_collector", version="1.0", schema_version="v1", category="collector"))
class LatencyCollector(ObservabilityModule):
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
        
    def record_latency(self, name: str, duration_sec: float) -> None:
        if self.buffer is None:
            return
            
        obs = MetricObservation(
            timestamp=Clock.now(),
            name=name,
            value=duration_sec,
            unit="sec"
        )
        self.buffer.add(obs)
