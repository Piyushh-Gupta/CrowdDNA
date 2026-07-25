import time
import threading
from crowdflow_dna.config import OPERATIONS_ALERT_THROTTLING_WINDOW_SEC

class AlertThrottler:
    def __init__(self):
        self.history = {}
        self.lock = threading.Lock()
        
    def is_throttled(self, alert_id: str) -> bool:
        now = time.time()
        with self.lock:
            if alert_id in self.history and (now - self.history[alert_id]) < OPERATIONS_ALERT_THROTTLING_WINDOW_SEC:
                return True
            self.history[alert_id] = now
            return False
