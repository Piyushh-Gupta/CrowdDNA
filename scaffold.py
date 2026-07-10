import os
import sys

base_dir = r"c:\Users\piyus\Downloads\CrowdDNA"

directories = [
    "crowdflow_dna/ingestion",
    "crowdflow_dna/detection",
    "crowdflow_dna/tracking",
    "crowdflow_dna/graph",
    "crowdflow_dna/model",
    "crowdflow_dna/rendering",
    "training",
    "tests",
    "configs",
    "notebooks",
    "assets",
    ".github/workflows"
]

files_to_create = [
    "crowdflow_dna/__init__.py",
    "crowdflow_dna/config.py",
    "crowdflow_dna/pipeline.py",
    "crowdflow_dna/errors.py",
    "crowdflow_dna/ingestion/__init__.py",
    "crowdflow_dna/ingestion/video_loader.py",
    "crowdflow_dna/detection/__init__.py",
    "crowdflow_dna/detection/detector.py",
    "crowdflow_dna/tracking/__init__.py",
    "crowdflow_dna/tracking/tracker.py",
    "crowdflow_dna/graph/__init__.py",
    "crowdflow_dna/graph/graph_builder.py",
    "crowdflow_dna/model/__init__.py",
    "crowdflow_dna/model/gnn_encoder.py",
    "crowdflow_dna/model/temporal_model.py",
    "crowdflow_dna/model/risk_classifier.py",
    "crowdflow_dna/rendering/__init__.py",
    "crowdflow_dna/rendering/video_renderer.py",
    "crowdflow_dna/rendering/timeline.py",
    "training/simulate_data.py",
    "training/prepare_datasets.py",
    "training/train_gnn.py",
    "training/export_onnx.py",
    "tests/test_ingestion.py",
    "tests/test_detection.py",
    "tests/test_tracking.py",
    "tests/test_graph.py",
    "tests/test_model.py",
    "tests/test_pipeline.py",
    "configs/default.yaml",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    ".gitignore",
    "README.md",
    ".github/workflows/ci.yml"
]

print("Creating directories...")
for d in directories:
    dir_path = os.path.join(base_dir, d)
    os.makedirs(dir_path, exist_ok=True)
    print(f"  Created {d}")

print("Creating empty files...")
for f in files_to_create:
    file_path = os.path.join(base_dir, f)
    with open(file_path, "a", encoding="utf-8") as file:
        pass
    print(f"  Created {f}")

print("Scaffolding complete!")
