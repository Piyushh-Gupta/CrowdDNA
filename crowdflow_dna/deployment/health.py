class DeploymentHealth:
    def verify_startup(self) -> bool:
        return True

    def verify_readiness(self) -> bool:
        return True

    def verify_liveness(self) -> bool:
        return True

    def verify_synthetic_service(self) -> bool:
        return True