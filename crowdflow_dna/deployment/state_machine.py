from enum import Enum, auto

class DeploymentState(Enum):
    PENDING = auto()
    VALIDATING = auto()
    BUILDING = auto()
    DEPLOYING = auto()
    VERIFYING = auto()
    ACTIVE = auto()
    DEPLOY_FAILED = auto()
    VERIFY_FAILED = auto()
    ROLLING_BACK = auto()
    ROLLED_BACK = auto()
    FAILED = auto()

class DeploymentLifecycle:
    def __init__(self):
        self.state = DeploymentState.PENDING
        
        self.valid_transitions = {
            DeploymentState.PENDING: [DeploymentState.VALIDATING],
            DeploymentState.VALIDATING: [DeploymentState.BUILDING, DeploymentState.FAILED],
            DeploymentState.BUILDING: [DeploymentState.DEPLOYING, DeploymentState.FAILED],
            DeploymentState.DEPLOYING: [DeploymentState.VERIFYING, DeploymentState.DEPLOY_FAILED],
            DeploymentState.VERIFYING: [DeploymentState.ACTIVE, DeploymentState.VERIFY_FAILED],
            DeploymentState.DEPLOY_FAILED: [DeploymentState.ROLLING_BACK, DeploymentState.FAILED],
            DeploymentState.VERIFY_FAILED: [DeploymentState.ROLLING_BACK, DeploymentState.FAILED],
            DeploymentState.ROLLING_BACK: [DeploymentState.ROLLED_BACK, DeploymentState.FAILED],
            DeploymentState.ACTIVE: [],
            DeploymentState.ROLLED_BACK: [],
            DeploymentState.FAILED: []
        }

    def transition(self, next_state: DeploymentState):
        if next_state not in self.valid_transitions[self.state]:
            raise ValueError(f"Illegal state transition from {self.state.name} to {next_state.name}")
        self.state = next_state