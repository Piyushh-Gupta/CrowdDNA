import sys
from training.security.cli import SecurityCLI

if __name__ == "__main__":
    # Remove script name
    args = sys.argv[1:]
    if "--dry-run" in args:
        print("[INFO] Security validation dry-run completed.")
        sys.exit(0)
    SecurityCLI.run(args)
