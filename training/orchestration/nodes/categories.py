from enum import Enum

class NodeCategory(Enum):
    TRAINING = "Training"
    EVALUATION = "Evaluation"
    VALIDATION = "Validation"
    REPORTING = "Reporting"
    UTILITY = "Utility"
    CLEANUP = "Cleanup"
    CHECKPOINT = "Checkpoint"
