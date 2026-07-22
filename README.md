# CrowdFlow DNA

**Graph-Based Crowd Interaction Modeling for Predictive Risk Classification**

CrowdFlow DNA is an independent portfolio project that analyzes uploaded crowd videos to produce risk-classified outputs. Moving beyond simple occupancy counting, the system detects individuals, constructs a dynamic interaction graph representing spatial and kinematic relationships, and applies a Graph Neural Network (GAT) combined with a temporal model (GRU) to classify localized crowd risk into discrete categories: **Safe**, **Congesting**, and **Critical**.

## Project Architecture (Pipe-and-Filter)
- **Ingestion**: Validates MP4/AVI uploads and extracts frames.
- **Detection**: Uses YOLOv8n to identify pedestrians.
- **Tracking**: Uses ByteTrack for persistent multi-object tracking.
- **Graph Construction**: Builds an interaction graph using proximity, relative velocity, and density gradients.
- **GNN/Temporal Inference**: A PyTorch GAT+GRU model (exported to ONNX) predicts risk classes.
- **Rendering & UI**: Gradio application hosted on Hugging Face Spaces provides the frontend, annotated video, and risk timeline.

## Development

AI-assisted development in this repository follows:

- AGENTS.md
- WORKFLOW.md
- CODING_STANDARDS.md

## Quickstart (Development)

### 1. Local Setup
```bash
# Clone the repository
git clone https://github.com/your-username/CrowdDNA.git
cd CrowdDNA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install PyTorch according to your hardware (CPU or CUDA)
# Example for CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Running Tests
```bash
pytest tests/ -v
```

### 3. Running the App
```bash
python app.py
```

## Team Structure & Development
This project is developed jointly by **Piyush Gupta** (AI & Data Lead) and **Aayushi Gupta** (Pipeline & Frontend Lead). 

Please refer to the `CrowdFlow_DNA_Team_Workflow_Guide_v5.docx` for complete repository integration workflows, Git branch strategies, and module interface contracts.

