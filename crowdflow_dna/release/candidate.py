import threading
from .exceptions import StateTransitionError

class CandidateManager:
    def __init__(self):
        self.state = "draft"
        self.lock = threading.Lock()
        
    def create(self):
        with self.lock:
            if self.state != "draft":
                raise StateTransitionError(f"Cannot create RC from state: {self.state}")
            self.state = "rc"
        
    def promote(self):
        with self.lock:
            if self.state != "rc":
                raise StateTransitionError(f"Cannot promote from state: {self.state}")
            self.state = "release"
        
    def reject(self):
        with self.lock:
            if self.state != "rc":
                raise StateTransitionError(f"Cannot reject from state: {self.state}")
            self.state = "rejected"
        
    def freeze(self):
        with self.lock:
            if self.state not in ("release", "rejected"):
                raise StateTransitionError(f"Cannot freeze from state: {self.state}")
            self.state = "frozen"
