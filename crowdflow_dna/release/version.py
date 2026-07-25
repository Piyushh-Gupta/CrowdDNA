import re
from .exceptions import VersionError

class VersionManager:
    # Basic semver regex matching major.minor.patch[-prerelease]
    SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?$")
    
    def __init__(self, current_version: str = "1.0.0"):
        if not self.SEMVER_PATTERN.match(current_version):
            raise VersionError(f"Invalid SemVer: {current_version}")
        self.current_version = current_version
        
    def bump(self, part: str) -> str:
        match = self.SEMVER_PATTERN.match(self.current_version)
        if not match:
            raise VersionError(f"Invalid SemVer: {self.current_version}")
            
        major, minor, patch, prerelease = match.groups()
        major, minor, patch = int(major), int(minor), int(patch)
        
        if part == "major":
            major += 1
            minor = 0
            patch = 0
            prerelease = None
        elif part == "minor":
            minor += 1
            patch = 0
            prerelease = None
        elif part == "patch":
            patch += 1
            prerelease = None
        elif part == "rc":
            if prerelease and prerelease.startswith("rc."):
                rc_num = int(prerelease.split(".")[1]) + 1
                prerelease = f"rc.{rc_num}"
            else:
                patch += 1
                prerelease = "rc.1"
        else:
            raise VersionError(f"Unknown bump part: {part}")
            
        ver = f"{major}.{minor}.{patch}"
        if prerelease:
            ver = f"{ver}-{prerelease}"
        return ver
