import requests
from requests.adapters import HTTPAdapter

class ConnectionManager:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=config.pool_connections,
            pool_maxsize=config.pool_maxsize,
            pool_block=True
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self.config.timeout)
        return self.session.request(method, url, **kwargs)
        
    def close(self):
        self.session.close()
