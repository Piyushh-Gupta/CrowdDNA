import os
import re
from typing import Dict, Any
from training.framework.configuration import ConfigurationBundle, ConfigValue

class ConfigurationMerger:
    def __init__(self):
        # Precedence: lowest index = lowest priority, highest index = highest priority
        self.layers = ['default', 'system', 'project', 'experiment', 'runtime', 'environment', 'cli']
        self._raw_layers: Dict[str, Dict[str, Any]] = {layer: {} for layer in self.layers}
        
    def add_layer(self, layer_name: str, config: Dict[str, Any]) -> None:
        if layer_name not in self.layers:
            raise ValueError(f"Unknown configuration layer: {layer_name}")
        self._raw_layers[layer_name] = config
        
    def merge(self, configuration_version: str = "1.0") -> ConfigurationBundle:
        merged_values: Dict[str, ConfigValue] = {}
        for layer in self.layers:
            layer_config = self._raw_layers[layer]
            for k, v in layer_config.items():
                merged_values[k] = ConfigValue(value=v, provenance=layer)
                
        return ConfigurationBundle(
            configuration_version=configuration_version,
            values=merged_values
        )
class ConfigurationResolver:
    @staticmethod
    def resolve(bundle: ConfigurationBundle) -> ConfigurationBundle:
        # Resolves environment variable interpolations like ${VAR_NAME}
        resolved_values = {}
        pattern = re.compile(r'\$\{([^}]+)\}')
        for k, cv in bundle.values.items():
            val = cv.value
            if isinstance(val, str):
                def repl(match):
                    var_name = match.group(1)
                    return os.environ.get(var_name, match.group(0))
                val = pattern.sub(repl, val)
            resolved_values[k] = ConfigValue(value=val, provenance=cv.provenance)
            
        return ConfigurationBundle(
            configuration_version=bundle.configuration_version,
            values=resolved_values
        )
