import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CrowdDNA Observability CLI")
    parser.add_argument("--action", choices=["monitor", "profile", "health", "report"], required=True, help="Action to perform")
    parser.add_argument("--session-id", default="default_session", help="Session ID")
    parser.add_argument("--output", default="experiments/observability_exports", help="Output directory")
    return parser.parse_args()
