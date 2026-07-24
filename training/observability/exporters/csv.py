import csv
import os
from typing import List
from training.observability.metadata import Observation, MetricObservation
from training.observability.registry import ObservabilityRegistry, ModuleMetadata, ObservabilityModule

@ObservabilityRegistry.register(ModuleMetadata(name="csv_exporter", version="1.0", schema_version="v1", category="exporter"))
class CSVExporter(ObservabilityModule):
    def __init__(self):
        self.output_dir = "experiments/observability_exports"
        
    def initialize(self, output_dir: str = None) -> None:
        if output_dir:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def start(self) -> None:
        pass
        
    def stop(self) -> None:
        pass
        
    def shutdown(self) -> None:
        pass
        
    def export(self, session_id: str, observations: List[Observation]) -> None:
        # Filter for MetricObservations for flat CSV
        metrics = [obs for obs in observations if isinstance(obs, MetricObservation)]
        if not metrics:
            return
            
        file_path = os.path.join(self.output_dir, f"session_{session_id}_metrics.csv")
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Name", "Value", "Unit"])
            for m in metrics:
                writer.writerow([m.timestamp, m.name, m.value, m.unit])
