import threading
from typing import Dict, Any

class IncidentManager:
    def __init__(self):
        self.incidents = {}
        self.lock = threading.Lock()
        self.counter = 0
        
    def create_incident(self, report: Dict[str, Any]) -> str:
        with self.lock:
            self.counter += 1
            inc_id = f"INC-{self.counter:04d}"
            self.incidents[inc_id] = report
            return inc_id
