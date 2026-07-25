import os

base_dir = "src"

dirs = [
    "config",
    "api",
    "providers",
    "routing",
    "stores",
    "theme",
    "components/ui",
    "components/layout",
    "components/feedback",
    "assets/icons",
    "assets/logos",
    "assets/illustrations",
    "assets/fonts",
    "hooks",
    "utils",
    "features/dashboard",
    "features/workflows",
    "features/inference",
    "features/explainability",
    "features/experiments",
    "features/observability",
    "features/reproducibility",
    "features/security",
    "features/settings",
]

feature_subdirs = ["components", "hooks", "pages", "api"]

for d in dirs:
    path = os.path.join(base_dir, d)
    os.makedirs(path, exist_ok=True)
    if d.startswith("features/"):
        for sub in feature_subdirs:
            os.makedirs(os.path.join(path, sub), exist_ok=True)

print("Scaffolded directories successfully.")
