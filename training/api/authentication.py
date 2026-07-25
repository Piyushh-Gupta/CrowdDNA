class ApiAuthentication:
    @staticmethod
    def extract_token(headers: dict) -> str:
        return headers.get("Authorization", "")
