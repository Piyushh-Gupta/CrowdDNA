import json
class Serializer:
    @staticmethod
    def to_json(obj):
        return json.dumps(obj)
    
    @staticmethod
    def from_json(data):
        return json.loads(data)
