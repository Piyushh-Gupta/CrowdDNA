class OpenApiGenerator:
    @staticmethod
    def generate() -> dict:
        return {"openapi": "3.0.0", "info": {"title": "CrowdDNA API", "version": "1.0.0"}}
