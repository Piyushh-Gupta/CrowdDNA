import time

class Clock:
    @staticmethod
    def now() -> float:
        return time.time()
        
    @staticmethod
    def perf_counter() -> float:
        return time.perf_counter()
