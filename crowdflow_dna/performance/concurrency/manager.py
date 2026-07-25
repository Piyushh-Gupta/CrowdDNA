import concurrent.futures
from typing import Optional

from crowdflow_dna.config import (
    PERFORMANCE_MAX_THREADS,
    PERFORMANCE_MAX_PROCESSES
)

class ConcurrencyManager:
    def __init__(self, max_threads: Optional[int] = None, max_processes: Optional[int] = None):
        self.max_threads = max_threads if max_threads is not None else PERFORMANCE_MAX_THREADS
        self.max_processes = max_processes if max_processes is not None else PERFORMANCE_MAX_PROCESSES
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_processes)

    def run_in_thread(self, fn, *args, **kwargs):
        return self.thread_pool.submit(fn, *args, **kwargs)

    def run_in_process(self, fn, *args, **kwargs):
        return self.process_pool.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True):
        self.thread_pool.shutdown(wait=wait)
        self.process_pool.shutdown(wait=wait)