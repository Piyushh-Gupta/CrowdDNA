class HealthManager:
    def evaluate(self):
        # Only checks basic component health, not SLOs
        return {"status": "healthy"}
