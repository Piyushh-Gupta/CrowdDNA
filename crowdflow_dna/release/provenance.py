import datetime

class ProvenanceGenerator:
    def generate(self, commit: str, builder_id: str, version: str, artifacts: list):
        return {
            "commit": commit,
            "builder": builder_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "version": version,
            "artifacts": sorted(artifacts)
        }
