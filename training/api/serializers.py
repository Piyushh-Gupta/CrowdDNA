from typing import Any
class BaseSerializer:
    @staticmethod
    def serialize(domain_obj: Any) -> Any:
        return str(domain_obj)
