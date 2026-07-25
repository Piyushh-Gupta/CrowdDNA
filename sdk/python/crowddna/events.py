import threading
import logging

class EventSystem:
    def __init__(self):
        self.hooks = {
            "on_request": [],
            "on_response": [],
            "on_retry": [],
            "on_stream_event": [],
            "on_error": []
        }
        self.lock = threading.Lock()
        
    def register(self, hook_name, callback):
        with self.lock:
            if hook_name in self.hooks:
                self.hooks[hook_name].append(callback)
            
    def trigger(self, hook_name, *args, **kwargs):
        callbacks = []
        with self.lock:
            if hook_name in self.hooks:
                callbacks = list(self.hooks[hook_name])
                
        for cb in callbacks:
            try:
                cb(*args, **kwargs)
            except Exception as e:
                logging.error(f"Event callback failed for {hook_name}: {e}")
