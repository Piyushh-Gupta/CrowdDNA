import json
import os
from typing import List
from dataclasses import asdict
from training.observability.metadata import Observation
from training.observability.registry import ObservabilityRegistry, ModuleMetadata, ObservabilityModule

@ObservabilityRegistry.register(ModuleMetadata(name="json_exporter", version="1.0", schema_version="v1", category="exporter"))
class JSONExporter(ObservabilityModule):
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
        if not observations:
            return
            
        file_path = os.path.join(self.output_dir, f"session_{session_id}.json")
        
        # We append to JSON line by line or rewrite. For simplicity, we just dump a list of dicts.
        # Handling Enums requires custom serialization.
        def _default(obj):
            if hasattr(obj, 'name') and hasattr(obj, 'value'): # simplistic enum catch
                return obj.name
            return str(obj)
            
        data = [asdict(obs) for obs in observations]
        
        with open(file_path, 'w') as f:
            json.dump(data, f, default=_default, indent=2, sort_keys=True)
