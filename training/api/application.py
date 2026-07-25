class Application:
    def __init__(self):
        self.started = False

    def start(self):
        self.started = True
        
    def stop(self):
        self.started = False
