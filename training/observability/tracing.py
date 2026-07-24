import uuid
import threading
from typing import ContextManager
from contextlib import contextmanager
from training.observability.metadata import TraceObservation
from training.observability.buffer import ObservationBuffer
from training.observability.clock import Clock

class TracingEngine:
    """Lightweight tracing engine for pipeline stages."""
    
    def __init__(self, buffer: ObservationBuffer):
        self._buffer = buffer
        self._local = threading.local()
        
    @contextmanager
    def span(self, name: str, metadata: dict = None) -> ContextManager[str]:
        """Context manager for tracing a specific execution block."""
        if not hasattr(self._local, "active_spans"):
            self._local.active_spans = []
            
        active_spans = self._local.active_spans
        
        trace_id = active_spans[0].trace_id if active_spans else str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        parent_span_id = active_spans[-1].span_id if active_spans else None
        
        # We need a temporary record to track parent-child locally
        class LocalSpan:
            def __init__(self, t, s):
                self.trace_id = t
                self.span_id = s
                
        local_span = LocalSpan(trace_id, span_id)
        active_spans.append(local_span)
        
        start_time = Clock.perf_counter()
        abs_start = Clock.now()
        
        try:
            yield span_id
        finally:
            duration = Clock.perf_counter() - start_time
            active_spans.pop()
            
            trace_obs = TraceObservation(
                timestamp=abs_start,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                name=name,
                duration_sec=duration,
                metadata=metadata or {}
            )
            self._buffer.add(trace_obs)
