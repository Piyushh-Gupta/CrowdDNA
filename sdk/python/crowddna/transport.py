from .connection import ConnectionManager
from .retry import RetryPolicy
from .exceptions import ServerError, AuthenticationError, AuthorizationError, RateLimitError

class Transport:
    def __init__(self, config, auth_provider, middleware, event_system):
        self.config = config
        self.auth = auth_provider
        self.middleware = middleware
        self.events = event_system
        self.connection = ConnectionManager(config)
        self.retry = RetryPolicy(config)
        
    def send(self, method, path, headers=None, **kwargs):
        headers = headers or {}
        if self.auth:
            headers.update(self.auth.get_auth_headers())
        
        url = f"{self.config.base_url}{path}"
        req = {"method": method, "url": url, "headers": headers, **kwargs}
        req = self.middleware.process_request(req)
        self.events.trigger("on_request", req)
        
        def _make_req():
            return self.connection.request(**req)
            
        resp = self.retry.execute(_make_req, req["method"], req["headers"])
        
        self.events.trigger("on_response", resp)
        resp = self.middleware.process_response(resp)
        
        if resp.status_code == 401:
            raise AuthenticationError("Unauthorized")
        elif resp.status_code == 403:
            raise AuthorizationError("Forbidden")
        elif resp.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        elif resp.status_code >= 500:
            raise ServerError("Server Error")
            
        return resp
