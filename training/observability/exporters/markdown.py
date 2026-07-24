import os
from typing import List
from training.observability.metadata import Observation, EventObservation, HealthObservation, AlertObservation
from training.observability.registry import ObservabilityRegistry, ModuleMetadata, ObservabilityModule

@ObservabilityRegistry.register(ModuleMetadata(name="markdown_exporter", version="1.0", schema_version="v1", category="exporter"))
class MarkdownExporter(ObservabilityModule):
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
            
        file_path = os.path.join(self.output_dir, f"session_{session_id}_report.md")
        
        lines = [f"# Observability Report (Session: {session_id})\n"]
        
        events = [o for o in observations if isinstance(o, EventObservation)]
        if events:
            lines.append("## Events")
            for e in events:
                lines.append(f"- **{e.priority.name}** | {e.name}: {e.message}")
            lines.append("")
            
        alerts = [o for o in observations if isinstance(o, AlertObservation)]
        if alerts:
            lines.append("## Alerts")
            for a in alerts:
                lines.append(f"- **{a.severity.name}** | {a.alert_name}: {a.message}")
            lines.append("")
            
        health = [o for o in observations if isinstance(o, HealthObservation)]
        if health:
            lines.append("## Health Status")
            for h in health:
                lines.append(f"- **{h.severity.name}** | {h.component}: {h.message}")
            lines.append("")
            
        with open(file_path, 'w') as f:
            f.write("\n".join(lines))
