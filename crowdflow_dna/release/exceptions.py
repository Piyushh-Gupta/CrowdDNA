class ReleaseError(Exception):
    pass

class VersionError(ReleaseError):
    pass

class StateTransitionError(ReleaseError):
    pass

class ChecksumError(ReleaseError):
    pass

class LicenseValidationError(ReleaseError):
    pass
