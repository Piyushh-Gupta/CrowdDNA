import json

class JSONExporter:
    def export(self, data):
        return json.dumps(data, sort_keys=True)

class MarkdownExporter:
    def export(self, data):
        return f"# Report\n{data}"

class CSVExporter:
    def export(self, data):
        if not data:
            return ""
        return ",".join(map(str, data.values()))
