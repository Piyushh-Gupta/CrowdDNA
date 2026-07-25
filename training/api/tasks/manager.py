import sys
import threading
from typing import Callable

class BackgroundTaskManager:
    @staticmethod
    def _run_with_error_handling(func: Callable, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            # Prevent swallowed exceptions in background threads
            sys.stderr.write(f"Background task failed: {e}\n")

    @staticmethod
    def dispatch(func: Callable, *args, **kwargs):
        thread = threading.Thread(target=BackgroundTaskManager._run_with_error_handling, args=(func,) + args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
