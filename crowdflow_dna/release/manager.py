from .candidate import CandidateManager
from .validation import ValidationManager

class ReleaseManager:
    def __init__(self):
        self.candidate_manager = CandidateManager()
        self.validation = ValidationManager()
        
    def orchestrate_release(self):
        self.candidate_manager.create()
        self.candidate_manager.promote()
        self.candidate_manager.freeze()
        return True
