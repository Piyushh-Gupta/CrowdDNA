from .configuration import Configuration
from .authentication import AuthenticationProvider
from .events import EventSystem
from .middleware import MiddlewareHooks
from .transport import Transport
from .services.workflow import WorkflowService
from .services.inference import InferenceService

class CrowdDNAClient:
    def __init__(self, config: Configuration, auth: AuthenticationProvider):
        self.config = config
        self.auth = auth
        self.events = EventSystem()
        self.middleware = MiddlewareHooks()
        self.transport = Transport(config, auth, self.middleware, self.events)
        
        self.workflows = WorkflowService(self.transport)
        self.inference = InferenceService()
