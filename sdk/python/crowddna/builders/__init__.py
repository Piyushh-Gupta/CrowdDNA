class ValidationError(Exception):
    pass

class RequestBuilder:
    def __init__(self):
        self.params = {}
        
    def build(self):
        return self.params.copy()

class WorkflowRequestBuilder(RequestBuilder):
    def with_id(self, w_id):
        self.params['id'] = w_id
        return self
        
    def build(self):
        if 'id' not in self.params:
            raise ValidationError("Workflow ID is required")
        return super().build()
