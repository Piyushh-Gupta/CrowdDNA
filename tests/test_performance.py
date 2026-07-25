import asyncio
from crowdflow_dna.performance.caching import CacheManager
from crowdflow_dna.performance.batching import DynamicBatcher
from crowdflow_dna.performance.concurrency import ConcurrencyManager
from crowdflow_dna.performance.optimization import OptimizationHooks
from crowdflow_dna.performance.benchmarks import BenchmarkRunner, BenchmarkBaseline

def test_cache_manager():
    cache = CacheManager(max_size=2, ttl_seconds=10)
    
    # Deterministic keys
    key1 = {"a": 1, "b": 2}
    key2 = {"b": 2, "a": 1}
    assert cache._generate_key(key1) == cache._generate_key(key2)
    
    cache.set(key1, "value")
    assert cache.get(key2) == "value"
    
    # Eviction
    cache.set({"k": 2}, "v2")
    cache.set({"k": 3}, "v3") # Should evict key1
    assert cache.get(key1) is None
    
    # Invalidation
    cache.invalidate({"k": 2})
    assert cache.get({"k": 2}) is None

def test_dynamic_batcher():
    async def run_batcher():
        batcher = DynamicBatcher(max_batch_size=2, batch_timeout_ms=10)
        
        async def mock_process(batch):
            return [f"processed_{i}" for i in batch]
            
        t1 = asyncio.create_task(batcher.process_item(1, mock_process))
        t2 = asyncio.create_task(batcher.process_item(2, mock_process))
        
        res1, res2 = await asyncio.gather(t1, t2)
        assert res1 == "processed_1"
        assert res2 == "processed_2"
    
    asyncio.run(run_batcher())

def test_concurrency_manager():
    mgr = ConcurrencyManager(max_threads=2, max_processes=2)
    def task():
        return 42
    
    fut = mgr.run_in_thread(task)
    assert fut.result() == 42
    mgr.shutdown()

def test_optimization_hooks():
    hooks = OptimizationHooks(pin_memory=True)
    cfg = hooks.optimize_dataloader({})
    assert cfg['pin_memory'] is True
    assert cfg['prefetch_factor'] == 2

def test_benchmark_runner():
    baseline = BenchmarkBaseline("sha", "1.0", "cpu", "linux", "3.11", "now", "hash")
    runner = BenchmarkRunner(baseline, warmup_iterations=2)
    
    def dummy():
        pass
        
    runner.run_benchmark("dummy", dummy, iterations=2)
    report = runner.get_report()
    assert len(report["results"]) == 1
    assert report["baseline"]["commit_sha"] == "sha"