from typing import Dict, Any
import json

class ConfigurationLoader:
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r') as f:
            return json.load(f)
            
    # Future TOML/YAML support would go here
