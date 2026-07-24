from typing import List
from training.observability.metadata import MonitoringSession, Observation, MetricObservation
from training.observability.buffer import ObservationBuffer
from training.observability.registry import ObservabilityRegistry
from training.observability.alerts import AlertEngine

class AggregationEngine:
    """Aggregates and routes observations from the buffer to exporters and alerting."""
    
    def __init__(self, buffer: ObservationBuffer, alert_engine: AlertEngine = None):
        self._buffer = buffer
        self._alert_engine = alert_engine
        # simple memory store of all history if needed, though usually handled by exporters
        self.history: List[Observation] = []
        
    def flush_and_process(self) -> List[Observation]:
        """Flushes the buffer and processes observations."""
        observations = self._buffer.flush()
        
        for obs in observations:
            self.history.append(obs)
            if self._alert_engine and isinstance(obs, MetricObservation):
                self._alert_engine.evaluate(obs)
                
        return observations

class ObservabilityEngine:
    """Main orchestration engine for the Observability framework."""
    
    def __init__(self, session: MonitoringSession, buffer_size: int = 10000):
        self.session = session
        self.buffer = ObservationBuffer(maxlen=buffer_size)
        self.alert_engine = AlertEngine(self.buffer)
        self.aggregator = AggregationEngine(self.buffer, self.alert_engine)
        self.active_modules = []
        
        # Initialize modules
        for name in session.context.enabled_collectors:
            mod_cls = ObservabilityRegistry.get_module(name)
            mod = mod_cls()
            mod.initialize(self.buffer)
            self.active_modules.append(mod)
            
    def start(self):
        for mod in self.active_modules:
            mod.start()
            
    def stop(self):
        for mod in self.active_modules:
            mod.stop()
            
    def shutdown(self):
        self.stop()
        for mod in self.active_modules:
            mod.shutdown()
        
    def process_tick(self):
        """Called periodically to collect metrics and flush buffers."""
        for mod in self.active_modules:
            if hasattr(mod, 'collect'):
                mod.collect()
        
        # Push to alerts and history
        # Exporters would typically be hooked here to write `flushed` observations to disk.
        flushed = self.aggregator.flush_and_process()
        return flushed
