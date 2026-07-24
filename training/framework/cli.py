import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CrowdDNA Framework CLI")
    parser.add_argument("--dry-run", action="store_true", help="Perform full discovery, resolution, validation, and graph construction without calling start()")
    return parser.parse_args()
