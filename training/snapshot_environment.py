import sys
import os

# Add root directory to python path if not present
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.reproducibility.cli import main

if __name__ == "__main__":
    main()
