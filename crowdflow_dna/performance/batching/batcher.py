import asyncio
from typing import Any, Callable, Optional

from crowdflow_dna.config import (
    PERFORMANCE_MAX_BATCH_SIZE,
    PERFORMANCE_BATCH_TIMEOUT_MS
)

class DynamicBatcher:
    def __init__(self, max_batch_size: Optional[int] = None, batch_timeout_ms: Optional[int] = None):
        self.max_batch_size = max_batch_size if max_batch_size is not None else PERFORMANCE_MAX_BATCH_SIZE
        self.batch_timeout_ms = batch_timeout_ms if batch_timeout_ms is not None else PERFORMANCE_BATCH_TIMEOUT_MS
        self.queue = []
        self.futures = []
        self._task = None
        self._lock = asyncio.Lock()

    async def _process_loop(self, process_fn: Callable):
        while True:
            await asyncio.sleep(self.batch_timeout_ms / 1000.0)
            async with self._lock:
                if self.queue:
                    await self._flush(process_fn)

    async def _flush(self, process_fn: Callable):
        batch = self.queue[:self.max_batch_size]
        futs = self.futures[:self.max_batch_size]
        
        self.queue = self.queue[self.max_batch_size:]
        self.futures = self.futures[self.max_batch_size:]
        
        if not batch:
            return

        try:
            results = await process_fn(batch)
            for f, res in zip(futs, results):
                if not f.done():
                    f.set_result(res)
        except Exception as e:
            for f in futs:
                if not f.done():
                    f.set_exception(e)

    async def process_item(self, item: Any, process_fn: Callable) -> Any:
        loop = asyncio.get_running_loop()
        f = loop.create_future()
        
        async with self._lock:
            self.queue.append(item)
            self.futures.append(f)

            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._process_loop(process_fn))

            if len(self.queue) >= self.max_batch_size:
                await self._flush(process_fn)

        return await f