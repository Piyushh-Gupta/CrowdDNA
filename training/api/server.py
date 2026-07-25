from training.api.application import Application

class Server:
    def __init__(self, app: Application):
        self.app = app

    def run(self):
        self.app.start()
