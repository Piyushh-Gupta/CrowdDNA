from crowdflow_dna.operations.context import OperationsState

class MaintenanceManager:
    def __init__(self, engine):
        self.engine = engine
        
    def request_maintenance(self):
        self.engine.transition_state(OperationsState.MAINTENANCE_REQUESTED)
        
    def start_drain(self):
        self.engine.transition_state(OperationsState.DRAINING)
        
    def enter_maintenance(self):
        self.engine.transition_state(OperationsState.MAINTENANCE)
        
    def recover(self):
        self.engine.transition_state(OperationsState.RECOVERING)
        self.engine.transition_state(OperationsState.NORMAL)
