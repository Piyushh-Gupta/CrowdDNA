from training.observability.registry import ObservabilityRegistry, ModuleMetadata, ObservabilityModule
from training.observability.metadata import MetricObservation
from training.observability.buffer import ObservationBuffer
from training.observability.clock import Clock

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

@ObservabilityRegistry.register(ModuleMetadata(name="gpu_collector", version="1.0", schema_version="v1", category="collector"))
class GPUCollector(ObservabilityModule):
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
        if self.buffer is None or not HAS_TORCH or not torch.cuda.is_available():
            return
            
        # Simplistic collection for GPU 0
        allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
        obs = MetricObservation(
            timestamp=Clock.now(),
            name="gpu_vram_allocated",
            value=allocated,
            unit="MB"
        )
        self.buffer.add(obs)
