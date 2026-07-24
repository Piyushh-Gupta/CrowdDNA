import time

class Clock:
    """Abstraction for timestamps and duration measurements."""
    
    @staticmethod
    def now() -> float:
        """Returns the current time in seconds since the Epoch."""
        return time.time()
        
    @staticmethod
    def monotonic() -> float:
        """Returns a monotonic clock time in seconds for measuring durations."""
        return time.monotonic()
        
    @staticmethod
    def perf_counter() -> float:
        """Returns a performance counter in seconds for precise measurements."""
        return time.perf_counter()
