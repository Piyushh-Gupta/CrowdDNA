# CrowdDNA Architectural Decision Log (ADR)

This document records important architectural decisions made during the development of CrowdDNA.

The purpose is to preserve the reasoning behind technical decisions so they are not repeatedly revisited by future contributors or AI assistants.

---

# ADR-001

Date:
YYYY-MM-DD

Title:
Synthetic Crowd Generation using PySocialForce

Status:
Accepted

Decision

PySocialForce will be used to generate synthetic pedestrian trajectories for dataset creation.

Reason

- Open source
- Social Force Model implementation
- Deterministic simulations
- Easy integration with Python
- Suitable for graph-based downstream learning

Alternatives Considered

- Menge
- Vadere
- Custom simulator

Rejected because they introduced unnecessary complexity or integration overhead.

---

# ADR-002

Date:
YYYY-MM-DD

Title:
Use Local NumPy RNG

Status:
Accepted

Decision

Use

np.random.default_rng(seed)

instead of

np.random.seed()

Reason

- Independent random generators
- Thread-safe
- Does not modify global RNG state
- Recommended NumPy API

---

# ADR-003

Date:
YYYY-MM-DD

Title:
Rule-Based Auto Labeling

Status:
Accepted

Decision

Synthetic trajectories are labeled using deterministic rules before model training.

Metrics

- Local Density
- Mean Speed
- Velocity Divergence

Priority

Critical

↓

Congesting

↓

Safe

Reason

Provides reproducible labels for supervised learning while keeping the pipeline explainable.

---

# ADR-004

Date:
YYYY-MM-DD

Title:
Lightweight CI Dependencies

Status:
Accepted

Decision

GitHub Actions installs requirements-ci.txt instead of the full requirements.txt.

Reason

- Faster CI
- Lower resource usage
- Heavy ML libraries installed only when required

---

# ADR-005

Date:
2026-07-21

Title:
Physical Coordinate Scaling for Proximity Graphs

Status:
Accepted

Decision

GraphBuilder and GraphDataset will operate on physical units (metres) rather than normalized coordinates [0, 1]. The `proximity_radius` is set to a physical distance (e.g., 2.0 metres). The JSON trajectory schema (`TrajectoryRecord.positions`) will store positions in metres without scene-relative normalization.

Reason

Phase 5 (`simulate_data.py`) generates positions in physical metres (scene is 20m x 20m). Normalizing positions strictly to [0, 1] loses physical meaning, makes proximity radius arbitrary, and couples the dataset layer to scene dimensions which are not serialized in the manifest. Relaxing `GraphBuilder` to accept physical units ensures edges accurately represent physical proximity regardless of scene size.

Alternatives Considered

- **Normalize positions in GraphDataset:** Would require serializing `scene_width` and `scene_height` in `manifest.json`.
- **Normalize in Phase 5:** PySocialForce works best in physical units; conversion would complicate downstream rendering and metric extraction.

Consequences

`GraphBuilder` no longer validates that `proximity_radius <= 1.0`. Any downstream components must be aware that node features (positions and speeds) are in physical units.

---

# ADR-006

Date:
2026-07-22

Title:
Fixed-Length Sequence Assumption for Temporal Encoder

Status:
Accepted

Decision

The TemporalEncoder and CrowdDNAModel currently assume fixed-length sequences across all trajectories in a batch (e.g., exactly 300 timesteps). Short sequences are zero-padded to `max_seq_len` internally within `CrowdDNAModel.forward()`. The GRU processes the padding tokens without masking or packing. 

Reason

Phase 5 synthetic trajectories are exactly 300 timesteps long. As long as sequences are fixed-length, the zero padding is never practically applied. Deferring packed sequence implementation keeps the current phase simple and focused.

Consequences

When variable-length trajectories are introduced, the zero-padding logic will corrupt the final GRU hidden state for any sequence shorter than `max_seq_len`. This is a known limitation that must be refactored using `torch.nn.utils.rnn.pack_padded_sequence` when variable-length data is supported.

---

# ADR Template

Date:

Title:

Status:

Decision

Reason

Alternatives Considered

Consequences