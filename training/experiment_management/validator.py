"""
Validator module for ensuring experiment directory consistency before loading.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ArtifactValidator:
    """Verifies internal consistency of an experiment directory."""

    @staticmethod
    def is_valid_experiment(exp_dir: Path) -> bool:
        """
        Validates if the given directory contains a complete and parseable experiment.
        
        Rules:
        - Must contain `metadata/experiment_metadata.json`
        - Must contain a checkpoint (e.g. `checkpoints/best.pt`)
        """
        if not exp_dir.is_dir():
            return False

        meta_path = exp_dir / "metadata" / "experiment_metadata.json"
        if not meta_path.exists():
            logger.debug(f"Skipping {exp_dir.name}: missing experiment_metadata.json")
            return False

        # Note: Depending on future ablation studies, we might relax the checkpoint requirement,
        # but for now we enforce it.
        checkpoints_dir = exp_dir / "checkpoints"
        if not checkpoints_dir.exists() or not any(checkpoints_dir.iterdir()):
            logger.debug(f"Skipping {exp_dir.name}: missing checkpoints")
            return False

        return True
