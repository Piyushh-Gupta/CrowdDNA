import sys

def main():
    if "--dry-run" in sys.argv:
        print("Dry run complete.")
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())
