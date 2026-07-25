import time

class StreamingClient:
    def __init__(self, transport, config):
        self.transport = transport
        self.config = config
        
    def listen(self, path, params=None):
        params = params or {}
        last_event_id = None
        headers = {}
        
        while True:
            if last_event_id:
                headers["Last-Event-ID"] = last_event_id
                
            try:
                resp = self.transport.send("GET", path, params=params, headers=headers, stream=True)
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode('utf-8')
                    if decoded.startswith("id:"):
                        last_event_id = decoded[3:].strip()
                    yield decoded
                break
            except Exception:
                time.sleep(self.config.streaming_reconnect_delay)
