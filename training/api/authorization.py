class ApiAuthorization:
    @staticmethod
    def authorize(context: dict, action: str, resource: str) -> bool:
        return True
