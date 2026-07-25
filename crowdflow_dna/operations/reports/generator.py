from crowdflow_dna.config import OPERATIONS_REPORT_LIMIT
class ReportGenerator:
    def __init__(self, exporters):
        self.exporters = exporters
        
    def generate(self, data, format="json"):
        # Truncate to limit
        if isinstance(data, dict):
            keys = sorted(data.keys())[:OPERATIONS_REPORT_LIMIT]
            data = {k: data[k] for k in keys}
            
        if format in self.exporters:
            return self.exporters[format].export(data)
        return data
