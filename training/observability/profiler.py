import cProfile
import pstats
import io
from typing import Optional

class ProfilerEngine:
    """Optional profiling layer to track deep CPU/memory execution paths."""
    
    def __init__(self):
        self._profiler: Optional[cProfile.Profile] = None
        
    def start(self):
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        
    def stop(self) -> str:
        if not self._profiler:
            return "Profiler was not running."
            
        self._profiler.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(self._profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions
        
        self._profiler = None
        return s.getvalue()
