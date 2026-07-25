from .score import CertificationScore
from ... import config

class CertificationGate:
    def evaluate(self, score: CertificationScore) -> bool:
        return score.overall >= config.SECURITY_MIN_CERTIFICATION_SCORE
