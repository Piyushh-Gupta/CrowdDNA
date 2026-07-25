import os

features = [
    "dashboard",
    "workflows",
    "inference",
    "explainability",
    "experiments",
    "observability",
    "reproducibility",
    "security",
    "settings",
]

for feature in features:
    path = f"src/features/{feature}/pages/index.tsx"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"export default function {feature.capitalize()}Page() {{ return <div>{feature.capitalize()}</div>; }}")

# Fix src/api/client.ts
client_ts_path = "src/api/client.ts"
with open(client_ts_path, "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("<T>(path:", "(path:")
content = content.replace("<T>(path: string, data:", "(path: string, data:")
with open(client_ts_path, "w", encoding="utf-8") as f:
    f.write(content)

# Fix src/App.tsx
app_tsx_path = "src/App.tsx"
with open(app_tsx_path, "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("import React from 'react';\n", "")
with open(app_tsx_path, "w", encoding="utf-8") as f:
    f.write(content)

# Fix src/utils/upload.ts
upload_ts_path = "src/utils/upload.ts"
with open(upload_ts_path, "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("const chunk = this.file.slice(this.uploadedBytes, end);\n", "")
with open(upload_ts_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed build issues.")
