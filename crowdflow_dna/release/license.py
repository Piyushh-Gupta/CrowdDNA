
class LicenseValidator:
    def __init__(self, allowlist: list, denylist: list):
        self.allowlist = set(allowlist)
        self.denylist = set(denylist)
        
    def validate(self, dependencies: list):
        report = {"compliant": True, "violations": []}
        
        # Deduplicate
        dep_dict = {}
        for d in dependencies:
            name = d.get("name")
            if name:
                dep_dict[name] = d
                
        for name in sorted(dep_dict.keys()):
            d = dep_dict[name]
            lic = d.get("license", "UNKNOWN")
            
            if lic in self.denylist:
                report["compliant"] = False
                report["violations"].append(f"{name} uses denied license {lic}")
            elif lic not in self.allowlist:
                report["compliant"] = False
                report["violations"].append(f"{name} uses unknown license {lic}")
                
        return report
