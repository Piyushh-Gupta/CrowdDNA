class MiddlewareHooks:
    def __init__(self):
        self.request_interceptors = []
        self.response_interceptors = []
        
    def add_request_interceptor(self, interceptor):
        self.request_interceptors.append(interceptor)
        
    def add_response_interceptor(self, interceptor):
        self.response_interceptors.append(interceptor)
        
    def process_request(self, request):
        for interceptor in self.request_interceptors:
            try:
                request = interceptor(request)
            except Exception:
                pass # Isolate middleware failures
        return request
        
    def process_response(self, response):
        for interceptor in self.response_interceptors:
            try:
                response = interceptor(response)
            except Exception:
                pass
        return response
