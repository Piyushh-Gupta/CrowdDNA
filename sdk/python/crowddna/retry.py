import time
import random
from .exceptions import TransportError

class RetryPolicy:
    def __init__(self, config):
        self.max_retries = config.max_retries
        self.base_delay = config.retry_delay_base
        
    def is_retryable(self, method, headers, status_code):
        if status_code in (429, 502, 503, 504):
            if method.upper() in ("GET", "HEAD") or "Idempotency-Key" in headers:
                return True
        return False
        
    def execute(self, func, method, headers):
        retries = 0
        while retries <= self.max_retries:
            try:
                resp = func()
                if not self.is_retryable(method, headers, resp.status_code):
                    return resp
                
                retries += 1
                if retries > self.max_retries:
                    return resp
                    
                # Handle Retry-After
                delay = self.base_delay * (2 ** retries) + random.uniform(0, 1)
                if resp.status_code == 429 and "Retry-After" in resp.headers:
                    try:
                        delay = float(resp.headers["Retry-After"])
                    except ValueError:
                        pass
                        
                time.sleep(delay)
            except Exception as e:
                if not self.is_retryable(method, headers, 503):
                    raise TransportError(str(e))
                retries += 1
                if retries > self.max_retries:
                    raise TransportError("Max retries exceeded")
                time.sleep(self.base_delay * (2 ** retries) + random.uniform(0, 1))
