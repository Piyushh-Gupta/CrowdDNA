import threading

class PublisherRegistry:
    def __init__(self):
        self.publishers = {}
        self.lock = threading.Lock()
        
    def register(self, name: str, publisher):
        with self.lock:
            self.publishers[name] = publisher
        
    def get(self, name: str):
        with self.lock:
            return self.publishers.get(name)
