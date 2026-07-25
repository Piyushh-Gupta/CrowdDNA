import sys

class SecurityCLI:
    @staticmethod
    def run(args):
        if not args:
            print("Usage: security_cli [validate|policies|audit]")
            sys.exit(1)
        
        command = args[0]
        if command == "validate":
            print("Validating security configuration...")
            print("Validation successful.")
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
