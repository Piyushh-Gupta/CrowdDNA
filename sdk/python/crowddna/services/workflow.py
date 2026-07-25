class WorkflowService:
    def __init__(self, transport):
        self.transport = transport
        
    def get_workflow(self, workflow_id):
        return self.transport.send("GET", f"/workflows/{workflow_id}")
