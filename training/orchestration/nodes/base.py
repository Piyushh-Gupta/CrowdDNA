from abc import ABC, abstractmethod
from training.orchestration.context import NodeContext
from training.orchestration.metadata import NodeResult
from training.orchestration.exceptions import OrchestrationException
from training.orchestration.nodes.categories import NodeCategory

class BaseNode(ABC):
    category: NodeCategory
    
    def prepare(self, context: NodeContext) -> None:
        """Optional pre-flight checks and setup."""
        pass
        
    @abstractmethod
    def execute(self, context: NodeContext) -> NodeResult:
        """Execute the primary ML logic idempotently."""
        pass
        
    def compensate(self, context: NodeContext, error: OrchestrationException) -> None:
        """Best-effort cleanup of artifacts if a workflow rollback is triggered."""
        pass
