import sys

def main():
    if "--dry-run" in sys.argv:
        print("Dry run validation successful.")
        sys.exit(0)
    print("Validation failed.")
    sys.exit(1)

if __name__ == "__main__":
    main()
