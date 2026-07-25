class SBOMGenerator:
    def generate(self, dependencies: list):
        # Deduplicate by name
        dep_dict = {}
        for d in dependencies:
            name = d.get("name")
            if name:
                dep_dict[name] = d
                
        components = []
        for name in sorted(dep_dict.keys()):
            d = dep_dict[name]
            components.append({
                "name": name,
                "version": d.get("version", "UNKNOWN"),
                "licenses": [{"license": {"id": d.get("license", "UNKNOWN")}}]
            })
            
        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "components": components
        }
