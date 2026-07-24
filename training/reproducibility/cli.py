import argparse
import logging
from pprint import pprint
from training.reproducibility.environment import capture_environment_snapshot
from training.reproducibility.manifest import ManifestManager
from training.reproducibility.validator import ReproducibilityValidator
from training.reproducibility.diff import ManifestDiffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="CrowdDNA Reproducibility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Snapshot
    subparsers.add_parser("snapshot", help="Capture current environment snapshot")
    
    # Validate
    validate_parser = subparsers.add_parser("validate", help="Validate a manifest against current environment")
    validate_parser.add_argument("manifest_path", type=str, help="Path to reproducibility_manifest.json")
    
    # Diff
    diff_parser = subparsers.add_parser("diff", help="Diff two manifests")
    diff_parser.add_argument("manifest1", type=str, help="Path to first manifest")
    diff_parser.add_argument("manifest2", type=str, help="Path to second manifest")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.command == "snapshot":
        snap = capture_environment_snapshot()
        logger.info("Environment Snapshot Captured:")
        import dataclasses
        pprint(dataclasses.asdict(snap))
        
    elif args.command == "validate":
        try:
            manifest = ManifestManager.load(args.manifest_path)
            current_env = capture_environment_snapshot()
            validator = ReproducibilityValidator()
            score, diagnostics = validator.validate(manifest, current_env)
            
            logger.info(f"Reproducibility Score: {score:.2f}/100")
            for diag in diagnostics:
                status = diag['status']
                logger.info(f"[{status}] {diag['validator']} (Severity: {diag['severity']}, Deduction: {diag['deduction']})")
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import sys
            sys.exit(1)
            
    elif args.command == "diff":
        try:
            manifest1 = ManifestManager.load(args.manifest1)
            manifest2 = ManifestManager.load(args.manifest2)
            
            diff_str = ManifestDiffer.diff_manifests(manifest1, manifest2)
            print(diff_str)
        except Exception as e:
            logger.error(f"Diff failed: {e}")
            import sys
            sys.exit(1)

if __name__ == "__main__":
    main()
