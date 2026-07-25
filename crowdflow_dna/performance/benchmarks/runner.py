import time
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkBaseline:
    commit_sha: str
    benchmark_version: str
    hardware_info: str
    os_info: str
    python_version: str
    timestamp: str
    config_hash: str

class BenchmarkRunner:
    def __init__(self, baseline: BenchmarkBaseline, warmup_iterations: int = 5):
        self.baseline = baseline
        self.warmup_iterations = warmup_iterations
        self.results = []

    def enforce_determinism(self):
        import random
        random.seed(42)

    def run_benchmark(self, name: str, fn, iterations: int = 10):
        self.enforce_determinism()
        
        # Warmup
        for _ in range(self.warmup_iterations):
            fn()
            
        start = time.time()
        for _ in range(iterations):
            fn()
        end = time.time()
        
        self.results.append({
            "name": name,
            "avg_latency": (end - start) / iterations,
            "total_time": end - start
        })

    def get_report(self):
        return {
            "baseline": asdict(self.baseline),
            "results": self.results
        }