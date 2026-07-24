"""
CrowdFlow DNA — Run Robustness Tests
====================================
"""

import sys
from training.robustness.cli import main

if __name__ == "__main__":
    sys.exit(main() or 0)
