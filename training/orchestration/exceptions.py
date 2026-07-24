class OrchestrationException(Exception):
    """Base exception for all orchestration errors."""
    pass

class WorkflowValidationError(OrchestrationException):
    pass

class WorkflowCycleError(WorkflowValidationError):
    pass

class ExecutionFailure(OrchestrationException):
    pass

class DependencyFailure(OrchestrationException):
    pass

class NodeExecutionError(OrchestrationException):
    pass

class NodeCompensationError(OrchestrationException):
    pass

class PolicyViolation(OrchestrationException):
    pass

class CheckpointCorruptionError(OrchestrationException):
    pass

class RecoveryFailure(OrchestrationException):
    pass

class ResourceLockAcquisitionError(OrchestrationException):
    pass

class WorkflowStateTransitionError(OrchestrationException):
    pass
