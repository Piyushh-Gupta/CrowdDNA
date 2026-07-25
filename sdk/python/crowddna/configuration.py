import os

class Configuration:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.timeout = 30
        self.max_retries = 3
        self.retry_delay_base = 1.0
        self.pool_connections = 10
        self.pool_maxsize = 100
        self.upload_chunk_size = 8 * 1024 * 1024
        self.download_buffer_size = 8192
        self.heartbeat_interval = 15
        self.polling_interval = 5
        self.pagination_default_limit = 50
        self.streaming_reconnect_delay = 3

class DevelopmentConfiguration(Configuration):
    def __init__(self):
        super().__init__(os.getenv("CROWDDNA_API_URL", "http://localhost:8000"))

class ProductionConfiguration(Configuration):
    def __init__(self):
        super().__init__(os.getenv("CROWDDNA_API_URL", "https://api.crowddna.io"))
        
class TestingConfiguration(Configuration):
    def __init__(self):
        super().__init__("http://testserver")
        self.max_retries = 0
