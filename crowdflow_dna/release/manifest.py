import json
import dataclasses

class ManifestGenerator:
    def generate(self, metadata):
        if dataclasses.is_dataclass(metadata):
            data = dataclasses.asdict(metadata)
        else:
            data = metadata
        return json.dumps(data, sort_keys=True, indent=2)
