# Implementation Backlog & Work Breakdown Structure (WBS)

## Epic 1: Data Engineering
- **Feature 1.1: Dataset Registry**
  - Task 1.1.1: CSV Loader
    - Subtask: Read trajectories
    - Subtask: Error handling for missing columns
  - Task 1.1.2: Trajectory Parser
    - Subtask: Extract spatio-temporal features
- **Feature 1.2: Preprocessing Pipeline**
  - Task 1.2.1: Normalization
    - Subtask: Min-max scaling
  - Task 1.2.2: Interpolation
    - Subtask: Spline interpolation for missing timesteps

## Epic 2: Graph Machine Learning
- **Feature 2.1: Graph Construction**
  - Task 2.1.1: Adjacency Matrices
    - Subtask: KNN distance calculation
- **Feature 2.2: Training Loop**
  - Task 2.2.1: Epoch Management
    - Subtask: Forward/backward passes

## Epic 3: Core API Services
- **Feature 3.1: Inference Engine**
  - Task 3.1.1: FastAPI Endpoints
    - Subtask: Request validation schemas

## Epic 4: SDK & Frontend Integration
- **Feature 4.1: Python SDK**
  - Task 4.1.1: Streaming Client
    - Subtask: WebSocket bindings
    - Blocked Until: Inference Engine Endpoints complete (Epic 3.1)
- **Feature 4.2: Dashboard UI**
  - Task 4.2.1: 2D Simulation Canvas
    - Subtask: Canvas rendering loop
    - Blocked Until: API Schemas Finalized (Epic 3.1)
