import os
from typing import List
import matplotlib.pyplot as plt
from training.observability.metadata import Observation, MetricObservation
from training.observability.registry import ObservabilityRegistry, ModuleMetadata, ObservabilityModule

@ObservabilityRegistry.register(ModuleMetadata(name="latency_plotter", version="1.0", schema_version="v1", category="plotter"))
class LatencyPlotter(ObservabilityModule):
    def __init__(self):
        self.output_dir = "experiments/observability_exports"
        
    def initialize(self, output_dir: str = None) -> None:
        if output_dir:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def start(self) -> None: pass
    def stop(self) -> None: pass
    def shutdown(self) -> None: pass
        
    def export(self, session_id: str, observations: List[Observation]) -> None:
        metrics = [o for o in observations if isinstance(o, MetricObservation) and "latency" in o.name]
        if not metrics:
            return
            
        times = [m.timestamp for m in metrics]
        values = [m.value for m in metrics]
        
        plt.figure()
        plt.plot(times, values, marker='o')
        plt.title('Latency Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Latency (s)')
        plt.savefig(os.path.join(self.output_dir, f"session_{session_id}_latency.png"))
        plt.close()
