import sys
from training.api.application import Application
from training.api.server import Server
from training.api.openapi import OpenApiGenerator

if __name__ == "__main__":
    args = sys.argv[1:]
    
    # Import endpoints to register them
    
    if "--dry-run" in args:
        print("[INFO] API dry-run completed.")
        print("[INFO] OpenAPI Spec generated:", OpenApiGenerator.generate())
        sys.exit(0)
    
    app = Application()
    server = Server(app)
    server.run()
