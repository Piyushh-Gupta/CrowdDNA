from typing import Callable, Any
from training.observability.engine import ObservabilityEngine
from training.observability.tracing import TracingEngine

class ObservabilityPipelineHooks:
    """Provides non-blocking hooks to integrate with InferenceRuntime and CrowdFlowPipeline."""
    
    def __init__(self, engine: ObservabilityEngine, tracer: TracingEngine):
        self._engine = engine
        self._tracer = tracer
        self._is_enabled = len(engine.active_modules) > 0
        
    def execute_with_tracing(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Wraps a function execution with a trace span."""
        if not self._is_enabled:
            return func(*args, **kwargs)
            
        with self._tracer.span(name):
            return func(*args, **kwargs)
