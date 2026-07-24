try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import shutil
from training.observability.metadata import HealthObservation, HealthSeverity
from training.observability.clock import Clock
from training.observability.buffer import ObservationBuffer

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class HealthMonitor:
    """Coordinates rapid, low-latency health checks."""
    
    def __init__(self, buffer: ObservationBuffer):
        self._buffer = buffer
        
    def check_all(self) -> None:
        """Executes all health checks and pushes them to the buffer."""
        self._check_disk_space()
        self._check_memory()
        self._check_gpu()
        
    def _check_disk_space(self) -> None:
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        if free_gb < 5.0:
            sev = HealthSeverity.ERROR if free_gb < 1.0 else HealthSeverity.WARNING
            msg = f"Low disk space: {free_gb:.1f}GB free"
            is_healthy = False
        else:
            sev = HealthSeverity.INFO
            msg = f"Disk space healthy: {free_gb:.1f}GB free"
            is_healthy = True
            
        self._buffer.add(HealthObservation(Clock.now(), "disk", sev, msg, is_healthy))

    def _check_memory(self) -> None:
        if not HAS_PSUTIL:
            self._buffer.add(HealthObservation(Clock.now(), "memory", HealthSeverity.WARNING, "psutil not available", False))
            return
            
        mem = psutil.virtual_memory()
        if mem.percent > 90.0:
            self._buffer.add(HealthObservation(Clock.now(), "memory", HealthSeverity.WARNING, f"High memory usage: {mem.percent}%", False))
        else:
            self._buffer.add(HealthObservation(Clock.now(), "memory", HealthSeverity.INFO, f"Memory usage normal: {mem.percent}%", True))
            
    def _check_gpu(self) -> None:
        if not HAS_TORCH or not torch.cuda.is_available():
            self._buffer.add(HealthObservation(Clock.now(), "gpu", HealthSeverity.WARNING, "GPU not available", False))
            return
            
        try:
            _ = torch.cuda.memory_allocated(0)
            self._buffer.add(HealthObservation(Clock.now(), "gpu", HealthSeverity.INFO, "GPU available and responding", True))
        except Exception as e:
            self._buffer.add(HealthObservation(Clock.now(), "gpu", HealthSeverity.ERROR, f"GPU error: {str(e)}", False))
