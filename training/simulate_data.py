"""
CrowdFlow DNA — Synthetic Crowd Data Generation
================================================
Module: training/simulate_data.py
Owner:  Piyush Gupta (AI & Data Lead)

Generates labeled synthetic crowd trajectory data using PySocialForce for
training the CrowdFlow DNA GAT+GRU risk classification model.

The module produces 300 JSON files (100 runs × 3 risk classes) and a
``manifest.json`` dataset index. The JSON schema is the authoritative
integration contract between this module and ``training/prepare_datasets.py``.

Downstream modules (prepare_datasets.py) must rely only on the serialized
JSON format and must not import any class or dataclass from this module
directly.

Usage (local / Colab):
    python training/simulate_data.py --config configs/default.yaml

SRD References:
    §4.6.6 Training Pipeline — Stage 1 (Dataset Preparation)
    §4.7   Dataset Planning
    §4.7.1 Data Sources
    §4.7.2 Labeling Approach
    §4.7.3 Splitting, Balancing, and Augmentation
    §4.7.4 Storage and Versioning
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("crowdflow.simulate_data")

# ---------------------------------------------------------------------------
# Simulation Constants  (Research-level labeling thresholds)
#
# These values are intentional research decisions and are NOT externalized to
# YAML. They will be tuned as simulation behaviour is observed during the
# data generation phase (SRD §4.7.2). All constants carry inline comments
# explaining their physical interpretation.
# ---------------------------------------------------------------------------

# -- Proximity --
PROXIMITY_RADIUS_M: float = 2.0
"""Radius (metres) used to compute each agent's local neighbourhood density."""

# -- Density thresholds (normalized: 0 = empty, 1 = maximum possible density) --
DENSITY_CONGESTING_THRESH: float = 0.35
"""Normalized local density above which a crowd region is considered compressing."""

DENSITY_CRITICAL_THRESH: float = 0.60
"""Normalized local density indicative of dangerous crowd compression."""

# -- Speed thresholds --
SPEED_CRITICAL_THRESH: float = 1.8
"""Mean agent speed (m/s) above which motion is classified as panic-level."""

# -- Velocity divergence thresholds (0 = perfectly aligned, 1 = fully chaotic) --
DIVERGENCE_CONGESTING_THRESH: float = 0.25
"""Divergence above which flow is becoming disorganized (Congesting boundary)."""

DIVERGENCE_CRITICAL_THRESH: float = 0.55
"""Divergence above which flow is classified as panic/chaotic (Critical boundary)."""

# -- Valid risk class labels (canonical, SRD §4.7.2) --
RISK_CLASSES: tuple[str, ...] = ("Safe", "Congesting", "Critical")

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """Parameters that fully define one PySocialForce simulation scenario.

    Constructed exclusively by ``ScenarioFactory``. This dataclass is an
    internal implementation detail; downstream modules must not depend on it.

    Attributes:
        scenario_name: Unique, snake_case identifier for this scenario
            (e.g. ``"safe_open_field"``).
        risk_class: Intended dominant risk class; one of ``"Safe"``,
            ``"Congesting"``, or ``"Critical"``.
        num_agents: Number of simulated pedestrians.
        scene_width: Horizontal extent of the simulation space in metres.
        scene_height: Vertical extent of the simulation space in metres.
        initial_speed: Mean initial walking speed assigned to agents (m/s).
        goal_spread: Standard deviation (metres) of the random goal position
            distribution sampled around predefined focal points.
        enable_panic: Whether to activate PySocialForce's panic/escape
            repulsion forces. Always ``True`` for Critical scenarios.
        description: Human-readable explanation stored in the manifest for
            traceability (SRD §4.7.4).
    """

    scenario_name: str
    risk_class: str
    num_agents: int
    scene_width: float
    scene_height: float
    initial_speed: float
    goal_spread: float
    enable_panic: bool
    description: str


@dataclass
class TrajectoryRecord:
    """Standardized, typed container for one complete labeled simulation run.

    This dataclass defines and internally validates the JSON schema that is
    the integration contract between ``simulate_data.py`` and
    ``prepare_datasets.py``. Downstream consumers must read the serialized
    JSON only — they must NOT import this class directly.

    JSON Schema (written by ``DatasetSerializer.save_record``):
    ::

        {
          "sequence_id":       str,
          "risk_class":        "Safe" | "Congesting" | "Critical",
          "scenario_name":     str,
          "num_agents":        int,
          "num_timesteps":     int,
          "generation_seed":   int,
          "label_distribution": {"Safe": int, "Congesting": int, "Critical": int},
          "positions":   [[[float, float], ...], ...],   // shape (T, N, 2)
          "velocities":  [[[float, float], ...], ...],   // shape (T, N, 2)
          "frame_labels": [str, ...]                     // length T
        }

    Attributes:
        sequence_id: Unique run identifier (e.g. ``"safe_open_field_run_042"``).
        risk_class: Scenario-level intended class. This identifies the
            scenario used to generate the sequence. It is scenario metadata
            only and is not guaranteed to match the majority of
            ``frame_labels``.
        scenario_name: Source scenario (matches ``ScenarioConfig.scenario_name``).
        num_agents: Number of agents tracked in this run.
        num_timesteps: Number of timesteps actually recorded.
        generation_seed: Random seed used; enables exact reproduction.
        label_distribution: Per-class timestep counts across the full run.
        positions: Agent positions per timestep; shape ``(T, N, 2)``,
            stored as nested Python lists for JSON compatibility.
        velocities: Agent velocities per timestep; shape ``(T, N, 2)``,
            stored as nested Python lists for JSON compatibility.
        frame_labels: Per-timestep risk label; length ``T``. These are the
            authoritative per-timestep labels used by downstream components.
    """

    sequence_id: str
    risk_class: str
    scenario_name: str
    num_agents: int
    num_timesteps: int
    generation_seed: int
    label_distribution: dict[str, int]
    positions: list[list[list[float]]]
    velocities: list[list[list[float]]]
    frame_labels: list[str]

    def validate(self) -> None:
        """Validates internal consistency of the record before serialization.

        Checks string fields for non-empty content, verifies ``risk_class``
        against the canonical ``RISK_CLASSES`` tuple, confirms that
        ``positions``, ``velocities``, and ``frame_labels`` all have length
        equal to ``num_timesteps``, that every timestep in ``positions`` and
        ``velocities`` contains exactly ``num_agents`` agents, that each
        agent entry is a two-element coordinate pair, that every label in
        ``frame_labels`` is a valid risk class, and that
        ``label_distribution`` keys match ``RISK_CLASSES`` exactly.

        Raises:
            ValueError: If any field violates its expected contract.
        """
        # -- String fields --
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty string.")
        if not self.scenario_name.strip():
            raise ValueError("scenario_name must be a non-empty string.")

        # -- Risk class --
        if self.risk_class not in RISK_CLASSES:
            raise ValueError(
                f"risk_class '{self.risk_class}' is not valid. "
                f"Expected one of {RISK_CLASSES}."
            )

        # -- Positive integer fields --
        if self.num_agents <= 0:
            raise ValueError(f"num_agents must be > 0, got {self.num_agents}.")
        if self.num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be > 0, got {self.num_timesteps}.")

        # -- positions shape: (T, N, 2) --
        T, N = self.num_timesteps, self.num_agents
        if len(self.positions) != T:
            raise ValueError(
                f"positions outer length {len(self.positions)} != num_timesteps {T}."
            )
        for t, frame in enumerate(self.positions):
            if len(frame) != N:
                raise ValueError(
                    f"positions[{t}] has {len(frame)} agents, expected {N}."
                )
            for n, coord in enumerate(frame):
                if len(coord) != 2:
                    raise ValueError(
                        f"positions[{t}][{n}] must be [x, y] (length 2), "
                        f"got length {len(coord)}."
                    )

        # -- velocities shape: (T, N, 2) --
        if len(self.velocities) != T:
            raise ValueError(
                f"velocities outer length {len(self.velocities)} != num_timesteps {T}."
            )
        for t, frame in enumerate(self.velocities):
            if len(frame) != N:
                raise ValueError(
                    f"velocities[{t}] has {len(frame)} agents, expected {N}."
                )
            for n, vec in enumerate(frame):
                if len(vec) != 2:
                    raise ValueError(
                        f"velocities[{t}][{n}] must be [vx, vy] (length 2), "
                        f"got length {len(vec)}."
                    )

        # -- frame_labels length and content --
        if len(self.frame_labels) != T:
            raise ValueError(
                f"frame_labels length {len(self.frame_labels)} != num_timesteps {T}."
            )
        invalid = [lbl for lbl in self.frame_labels if lbl not in RISK_CLASSES]
        if invalid:
            raise ValueError(
                f"frame_labels contains invalid labels: {set(invalid)}. "
                f"Expected values from {RISK_CLASSES}."
            )

        # -- label_distribution keys --
        if set(self.label_distribution.keys()) != set(RISK_CLASSES):
            raise ValueError(
                f"label_distribution must have exactly the keys {RISK_CLASSES}, "
                f"got {set(self.label_distribution.keys())}."
            )
        if any(v < 0 for v in self.label_distribution.values()):
            raise ValueError("label_distribution values must all be >= 0.")
        total = sum(self.label_distribution.values())
        if total != T:
            raise ValueError(
                f"label_distribution total {total} != num_timesteps {T}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the record to a JSON-serializable plain Python dictionary.

        All numeric values in ``positions`` and ``velocities`` are rounded to
        six decimal places to avoid floating-point noise in the output files.
        Positions and velocities are stored as nested Python lists (not NumPy
        arrays) to ensure compatibility with ``json.dump``.

        Returns:
            A dictionary whose structure matches the JSON schema documented
            in the class docstring. All values are JSON-serializable (str,
            int, float, list, dict).
        """
        return {
            "sequence_id": self.sequence_id,
            "risk_class": self.risk_class,
            "scenario_name": self.scenario_name,
            "num_agents": self.num_agents,
            "num_timesteps": self.num_timesteps,
            "generation_seed": self.generation_seed,
            "label_distribution": dict(self.label_distribution),
            "positions": [
                [
                    [round(coord, 6) for coord in agent]
                    for agent in frame
                ]
                for frame in self.positions
            ],
            "velocities": [
                [
                    [round(val, 6) for val in agent]
                    for agent in frame
                ]
                for frame in self.velocities
            ],
            "frame_labels": list(self.frame_labels),
        }


# ---------------------------------------------------------------------------
# ScenarioFactory
# ---------------------------------------------------------------------------


class ScenarioFactory:
    """Builds the canonical scenario configurations for all three risk classes.

    Each ``build_*`` method returns a single ``ScenarioConfig`` that
    represents the canonical crowd scenario for that risk class. All
    variability between runs of the same class is introduced via random
    seed variation in ``SimulationRunner``, not by having multiple configs.
    """

    @staticmethod
    def build_safe() -> ScenarioConfig:
        """Builds the Safe scenario: open-field uniform walking.

        Agents are spread across the scene with dispersed goals and low
        walking speed, representing normal low-density pedestrian flow.

        Returns:
            ScenarioConfig for the Safe crowd scenario.
        """
        return ScenarioConfig(
            scenario_name="safe_open_field",
            risk_class="Safe",
            num_agents=20,
            scene_width=20.0,
            scene_height=20.0,
            initial_speed=0.8,
            goal_spread=5.0,
            enable_panic=False,
            description="Open-field uniform walking with dispersed goals and low density."
        )

    @staticmethod
    def build_congesting() -> ScenarioConfig:
        """Builds the Congesting scenario: bottleneck compression.

        Agents funnel toward a narrow passage, causing increasing local
        density and mild flow disruption without active panic.

        Returns:
            ScenarioConfig for the Congesting crowd scenario.
        """
        return ScenarioConfig(
            scenario_name="congesting_bottleneck",
            risk_class="Congesting",
            num_agents=35,
            scene_width=20.0,
            scene_height=20.0,
            initial_speed=1.2,
            goal_spread=2.0,
            enable_panic=False,
            description="Agents funneling through a narrow bottleneck, causing compression."
        )

    @staticmethod
    def build_critical() -> ScenarioConfig:
        """Builds the Critical scenario: panic escape.

        Agents flee outward from a central point at high speed with
        divergent goals, simulating a stampede/evacuation event.
        ``enable_panic`` is set to ``True`` to activate PySocialForce
        escape repulsion forces.

        Returns:
            ScenarioConfig for the Critical crowd scenario.
        """
        return ScenarioConfig(
            scenario_name="critical_panic_escape",
            risk_class="Critical",
            num_agents=50,
            scene_width=20.0,
            scene_height=20.0,
            initial_speed=2.0,
            goal_spread=10.0,
            enable_panic=True,
            description="Panic escape from a central point with high speed and divergence."
        )

    @staticmethod
    def build_all() -> list[ScenarioConfig]:
        """Returns all three canonical scenario configurations.

        Order: [Safe, Congesting, Critical].

        Returns:
            List of ScenarioConfig, one per risk class.
        """
        return [
            ScenarioFactory.build_safe(),
            ScenarioFactory.build_congesting(),
            ScenarioFactory.build_critical(),
        ]


# ---------------------------------------------------------------------------
# SimulationRunner
# ---------------------------------------------------------------------------


class SimulationRunner:
    """Executes a single PySocialForce simulation run given a ScenarioConfig.

    Each call to ``run()`` is fully deterministic given the same seed,
    enabling exact reproducibility of any generated sequence (SRD §4.7.4).

    PySocialForce state format (per agent, 6 columns):
        ``[px, py, vx, vy, gx, gy]``
    where ``(px, py)`` is position, ``(vx, vy)`` is velocity, and
    ``(gx, gy)`` is the goal position.

    After stepping the simulator ``num_timesteps`` times, positions and
    velocities are read from the accumulated ``peds.get_states()`` buffer and
    returned as two ``float32`` NumPy arrays of shape
    ``(num_timesteps, N, 2)``.
    """

    # Congesting scenario: narrow passage occupies the horizontal centre of
    # the scene. Agents start on one side and must squeeze through.
    # Obstacle format for PySocialForce: [x_start, x_end, y_start, y_end]
    _BOTTLENECK_GAP_WIDTH: float = 2.0   # metres, width of the passage opening
    _BOTTLENECK_WALL_X: float = 10.0     # x-position of the wall (scene centre)

    def run(
        self,
        config: ScenarioConfig,
        seed: int,
        num_timesteps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs one PySocialForce simulation and returns trajectory arrays.

        Seeds NumPy before construction so all random choices (initial
        positions, goal sampling) are reproducible.

        Args:
            config: Scenario configuration defining agent count, scene
                geometry, speed, panic mode, etc.
            seed: Integer random seed for this run. Typically
                ``seed_base + run_index``.
            num_timesteps: Number of simulation steps to execute.

        Returns:
            A tuple ``(positions, velocities)`` where:
                - ``positions``  has shape ``(num_timesteps, N, 2)``
                - ``velocities`` has shape ``(num_timesteps, N, 2)``
            Both are ``np.ndarray`` of dtype ``float32``.

        Raises:
            RuntimeError: If PySocialForce fails to initialize or step.
        """
        import pysocialforce as psf  # deferred import — not available at CI lint time

        # Use a local Generator so this call never modifies NumPy's global
        # RNG state, making parallel or sequenced runs fully independent.
        rng = np.random.default_rng(seed)

        initial_state = self._build_initial_state(config, rng)
        obstacles = self._build_obstacles(config)

        try:
            sim = psf.Simulator(
                state=initial_state,
                groups=None,
                obstacles=obstacles,
            )
        except Exception as exc:
            raise RuntimeError(
                f"PySocialForce failed to initialize for scenario "
                f"'{config.scenario_name}' (seed={seed}): {exc}"
            ) from exc

        try:
            for _ in range(num_timesteps):
                sim.step()
        except Exception as exc:
            raise RuntimeError(
                f"PySocialForce step failed for scenario "
                f"'{config.scenario_name}' (seed={seed}): {exc}"
            ) from exc

        # get_states() returns shape (T+1, N, 7); the +1 is the initial state
        # recorded before the first step. We drop timestep 0 (pre-simulation)
        # and retain exactly num_timesteps rows.
        all_states, _ = sim.peds.get_states()
        # Slice to (num_timesteps, N, 7), then extract pos and vel columns
        states = all_states[1: num_timesteps + 1]  # shape (T, N, 7)
        positions = states[:, :, 0:2].astype(np.float32)   # (T, N, 2)
        velocities = states[:, :, 2:4].astype(np.float32)  # (T, N, 2)

        return positions, velocities

    def _build_initial_state(
        self,
        config: ScenarioConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Samples random initial positions and goal positions for all agents.

        For the Safe scenario agents are uniformly distributed across the
        scene with goals sampled near the opposite side, producing natural
        crossing flows.

        For the Congesting scenario agents start on the left half of the
        scene; goals are clustered tightly on the right side, forcing all
        agents through the bottleneck obstacle.

        For the Critical scenario agents start clustered at the scene centre
        and flee outward in all directions (panic escape). Goal positions are
        sampled near the scene boundary.

        Initial velocity is set to ``initial_speed`` along the direction from
        each agent's start position to its goal, so agents begin moving
        immediately on the first step.

        Args:
            config: Active scenario configuration.
            rng: Local NumPy Generator instance created in ``run()``.
                Using a local generator avoids modifying NumPy's global
                RNG state.

        Returns:
            NumPy array of shape ``(N, 6)`` with columns
            ``[px, py, vx, vy, gx, gy]``.
        """
        N = config.num_agents
        W, H = config.scene_width, config.scene_height
        speed = config.initial_speed
        spread = config.goal_spread

        if config.risk_class == "Safe":
            # Agents distributed uniformly; goals scattered on the opposite half
            px = rng.uniform(0.5, W * 0.45, size=N)
            py = rng.uniform(0.5, H - 0.5, size=N)
            gx = rng.uniform(W * 0.55, W - 0.5, size=N)
            gy = rng.uniform(0.5, H - 0.5, size=N)

        elif config.risk_class == "Congesting":
            # Agents on the left; goals clustered at the right side exit point
            px = rng.uniform(0.5, W * 0.4, size=N)
            py = rng.uniform(0.5, H - 0.5, size=N)
            goal_centre_x = W - 1.0
            goal_centre_y = H / 2.0
            gx = rng.normal(goal_centre_x, spread * 0.5, size=N)
            gy = rng.normal(goal_centre_y, spread * 0.5, size=N)

        else:  # Critical — panic escape from centre
            centre_x, centre_y = W / 2.0, H / 2.0
            # Agents clustered around the centre
            px = rng.normal(centre_x, 1.5, size=N)
            py = rng.normal(centre_y, 1.5, size=N)
            # Goals near the scene boundary in all directions
            angles = rng.uniform(0, 2 * np.pi, size=N)
            radius = rng.uniform(W * 0.4, W * 0.5, size=N)
            gx = centre_x + radius * np.cos(angles)
            gy = centre_y + radius * np.sin(angles)

        # Clip positions and goals to valid scene bounds with a small margin
        margin = 0.3
        px = np.clip(px, margin, W - margin)
        py = np.clip(py, margin, H - margin)
        gx = np.clip(gx, margin, W - margin)
        gy = np.clip(gy, margin, H - margin)

        # Compute unit direction from start to goal and apply initial speed
        dx, dy = gx - px, gy - py
        dist = np.hypot(dx, dy)
        dist = np.where(dist < 1e-6, 1e-6, dist)   # avoid zero-division
        vx = speed * dx / dist
        vy = speed * dy / dist

        state = np.column_stack([px, py, vx, vy, gx, gy])  # (N, 6)
        return state.astype(np.float64)

    def _build_obstacles(
        self,
        config: ScenarioConfig,
    ) -> list[list[float]] | None:
        """Constructs PySocialForce obstacle definitions from scenario config.

        Only the Congesting scenario uses obstacles (a wall with a narrow
        central passage). Safe and Critical scenarios return ``None``.

        Obstacle format expected by PySocialForce:
            ``[x_start, x_end, y_start, y_end]`` — axis-aligned line segments.

        Args:
            config: Active scenario configuration.

        Returns:
            A list of ``[x_start, x_end, y_start, y_end]`` obstacle
            definitions for the Congesting scenario, or ``None`` for all
            other scenarios.
        """
        if config.risk_class != "Congesting":
            return None

        H = config.scene_height
        x = self._BOTTLENECK_WALL_X
        gap = self._BOTTLENECK_GAP_WIDTH
        gap_start = (H / 2.0) - (gap / 2.0)
        gap_end = (H / 2.0) + (gap / 2.0)

        # Wall below the gap
        lower_wall = [x, x, 0.0, gap_start]
        # Wall above the gap
        upper_wall = [x, x, gap_end, H]

        return [lower_wall, upper_wall]


# ---------------------------------------------------------------------------
# AutoLabeler
# ---------------------------------------------------------------------------


class AutoLabeler:
    """Assigns a per-timestep risk label to a raw trajectory sequence.

    Labels are derived from crowd metrics computed at each timestep using
    the module-level ``SimulationConstants``. No ground truth from the
    simulation engine is used directly — labels emerge from the observed
    kinematic state of the crowd.

    The labeling rules are priority-ordered: Critical is evaluated before
    Congesting before Safe, ensuring danger classes are never suppressed.

    All methods are deterministic: given identical inputs they always
    produce identical outputs. No random state is used.
    """

    def label_sequence(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        config: ScenarioConfig,
    ) -> list[str]:
        """Labels all timesteps in a trajectory sequence.

        Iterates over each timestep, computes crowd metrics via
        ``_compute_metrics``, and assigns a risk label via ``_apply_rules``.

        Args:
            positions: Agent positions of shape ``(T, N, 2)``.
            velocities: Agent velocities of shape ``(T, N, 2)``.
            config: Scenario config; ``enable_panic`` flag is forwarded
                to ``_apply_rules`` to lower the Critical threshold.

        Returns:
            List of risk label strings of length ``T``, each element being
            one of ``"Safe"``, ``"Congesting"``, or ``"Critical"``.
        """
        T = positions.shape[0]
        labels: list[str] = []
        for t in range(T):
            metrics = self._compute_metrics(positions[t], velocities[t])
            label = self._apply_rules(metrics, config.enable_panic)
            labels.append(label)
        return labels

    def _compute_metrics(
        self,
        pos_t: np.ndarray,
        vel_t: np.ndarray,
    ) -> dict[str, float]:
        """Computes crowd-level kinematic metrics for a single timestep.

        **mean_local_density**: For every agent, counts how many other agents
        lie within ``PROXIMITY_RADIUS_M`` metres. The per-agent count is
        averaged across all agents and normalized by the maximum possible
        count ``(N - 1)`` so the result lies in ``[0, 1]``. Returns ``0.0``
        when there is only one agent.

        **mean_speed**: Mean L2 norm of velocity vectors across all ``N``
        agents in metres per second.

        **velocity_divergence**: Mean cosine *dissimilarity*
        ``(1 - cosine_similarity)`` computed over all unique agent pairs.
        A value of ``0`` means all agents move in exactly the same direction;
        a value of ``1`` means agents are moving in perfectly opposite
        directions on average. Returns ``0.0`` when fewer than two agents
        have non-zero velocity.

        Args:
            pos_t: Agent positions at timestep t; shape ``(N, 2)``.
            vel_t: Agent velocities at timestep t; shape ``(N, 2)``.

        Returns:
            Dictionary with float values for keys ``"mean_local_density"``,
            ``"mean_speed"``, and ``"velocity_divergence"``.
        """
        N = pos_t.shape[0]

        # -- mean_local_density --
        if N <= 1:
            mean_local_density = 0.0
        else:
            # Pairwise squared distances via broadcasting: (N, 1, 2) - (1, N, 2)
            diff = pos_t[:, np.newaxis, :] - pos_t[np.newaxis, :, :]  # (N, N, 2)
            sq_dist = (diff ** 2).sum(axis=2)                          # (N, N)
            # Neighbour count per agent (exclude self by checking > 0 dist)
            within_radius = (sq_dist < PROXIMITY_RADIUS_M ** 2) & (sq_dist > 0.0)
            neighbour_counts = within_radius.sum(axis=1).astype(float)  # (N,)
            mean_local_density = float(neighbour_counts.mean() / (N - 1))

        # -- mean_speed --
        speeds = np.linalg.norm(vel_t, axis=1)   # (N,)
        mean_speed = float(speeds.mean())

        # -- velocity_divergence --
        vel_norms = speeds                         # reuse (N,)
        moving = vel_norms > 1e-8                  # mask of agents with non-zero velocity
        n_moving = int(moving.sum())

        if n_moving < 2:
            velocity_divergence = 0.0
        else:
            v_moving = vel_t[moving]               # (M, 2)
            norms_moving = vel_norms[moving]       # (M,)
            unit_v = v_moving / norms_moving[:, np.newaxis]  # (M, 2)
            # Cosine similarity matrix: (M, M) via dot product of unit vectors
            cos_sim = unit_v @ unit_v.T            # (M, M), values in [-1, 1]
            # Dissimilarity for all unique pairs (upper triangle, excluding diag)
            i_upper, j_upper = np.triu_indices(n_moving, k=1)
            pair_dissimilarity = 1.0 - cos_sim[i_upper, j_upper]
            velocity_divergence = float(pair_dissimilarity.mean())

        return {
            "mean_local_density": mean_local_density,
            "mean_speed": mean_speed,
            "velocity_divergence": velocity_divergence,
        }

    def _apply_rules(
        self,
        metrics: dict[str, float],
        enable_panic: bool,
    ) -> str:
        """Applies priority-ordered labeling rules to a metrics snapshot.

        Rules are evaluated highest-priority first so that Critical is never
        overridden by a Congesting or Safe condition.

        Priority 1 — **Critical (panic path)**: triggered when ``enable_panic``
        is ``True`` *and* either the divergence or speed exceeds its Critical
        threshold. Reflects PySocialForce's active panic/escape forces.

        Priority 2 — **Critical (density path)**: triggered when both density
        and divergence are above their Critical thresholds. Captures genuine
        high-density chaotic situations regardless of the panic flag.

        Priority 3 — **Congesting**: triggered when density *or* divergence
        exceeds its Congesting threshold, indicating compression or flow
        disruption without full panic.

        Priority 4 — **Safe**: default when none of the above conditions hold.

        Args:
            metrics: Output of ``_compute_metrics`` for one timestep.
            enable_panic: Whether the active scenario has panic forces
                enabled. Used to lower the Critical classification bar.

        Returns:
            One of ``"Safe"``, ``"Congesting"``, or ``"Critical"``.
        """
        density = metrics["mean_local_density"]
        speed = metrics["mean_speed"]
        divergence = metrics["velocity_divergence"]

        # Priority 1: panic-mode Critical
        if enable_panic and (
            divergence > DIVERGENCE_CRITICAL_THRESH
            or speed > SPEED_CRITICAL_THRESH
        ):
            return "Critical"

        # Priority 2: density-driven Critical
        if (
            density > DENSITY_CRITICAL_THRESH
            and divergence > DIVERGENCE_CONGESTING_THRESH
        ):
            return "Critical"

        # Priority 3: Congesting
        if (
            density > DENSITY_CONGESTING_THRESH
            or divergence > DIVERGENCE_CONGESTING_THRESH
        ):
            return "Congesting"

        # Priority 4: Safe (default)
        return "Safe"

    @staticmethod
    def compute_label_distribution(labels: list[str]) -> dict[str, int]:
        """Counts occurrences of each risk label in a labeled sequence.

        All three canonical risk classes are always present as keys even if
        their count is zero, so downstream consumers can rely on a fixed
        dictionary structure without guarding against missing keys.

        Args:
            labels: List of risk label strings of length T. Every element
                must be one of the values in ``RISK_CLASSES``.

        Returns:
            Dictionary mapping each risk class to its timestep count,
            e.g. ``{"Safe": 280, "Congesting": 15, "Critical": 5}``.
            Keys are always exactly the three values in ``RISK_CLASSES``.
        """
        distribution: dict[str, int] = {cls: 0 for cls in RISK_CLASSES}
        for label in labels:
            distribution[label] += 1
        return distribution


# ---------------------------------------------------------------------------
# DatasetSerializer
# ---------------------------------------------------------------------------


class DatasetSerializer:
    """Writes TrajectoryRecord objects to JSON and maintains manifest.json.

    The JSON output of this class is the authoritative integration contract
    between simulate_data.py and prepare_datasets.py. The schema is defined
    by the ``TrajectoryRecord.to_dict()`` method.

    Each call to ``save_record`` writes exactly one ``<sequence_id>.json``
    file. After all records are saved, ``write_manifest`` writes a single
    ``manifest.json`` that indexes the entire dataset. Both methods are
    deterministic: identical inputs always produce identical byte-for-byte
    output.
    """

    def save_record(
        self,
        record: TrajectoryRecord,
        output_dir: str,
    ) -> str:
        """Validates and serializes one TrajectoryRecord to a JSON file.

        The output filename is ``<sequence_id>.json``. The file is written
        with 2-space indentation and sorted keys so that diffs are
        human-readable and output is deterministic regardless of insertion
        order.

        Args:
            record: A ``TrajectoryRecord`` instance. ``validate()`` is
                called before writing; any validation failure raises
                ``ValueError`` without creating the file.
            output_dir: Directory path where the JSON file will be written.
                Created (including intermediate directories) if it does not
                already exist.

        Returns:
            Absolute path of the written JSON file as a string.

        Raises:
            ValueError: If ``record.validate()`` fails.
            IOError: If the file cannot be written to ``output_dir``.
        """
        record.validate()

        out_path = os.path.join(output_dir, f"{record.sequence_id}.json")
        os.makedirs(output_dir, exist_ok=True)

        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(record.to_dict(), fh, indent=2, sort_keys=True)
                fh.write("\n")  # POSIX-compliant trailing newline
        except OSError as exc:
            raise IOError(
                f"Failed to write record '{record.sequence_id}' "
                f"to '{out_path}': {exc}"
            ) from exc

        logger.debug("Saved record '%s' → %s", record.sequence_id, out_path)
        return os.path.abspath(out_path)

    def write_manifest(
        self,
        records: list[TrajectoryRecord],
        output_dir: str,
    ) -> None:
        """Writes a manifest.json index for an entire dataset.

        The manifest is a JSON array sorted by ``sequence_id`` so that
        entries are always in a stable, deterministic order. Each entry
        contains lightweight metadata only — no trajectory arrays — so the
        file remains small regardless of dataset size.

        Manifest entry schema::

            {
              "sequence_id":        str,
              "risk_class":         str,
              "scenario_name":      str,
              "num_agents":         int,
              "num_timesteps":      int,
              "generation_seed":    int,
              "label_distribution": {"Safe": int, "Congesting": int, "Critical": int},
              "file_path":          str   // relative path from output_dir
            }

        Args:
            records: All ``TrajectoryRecord`` instances produced in this
                generation run. May be empty, in which case an empty array
                is written.
            output_dir: Directory where ``manifest.json`` will be written.
                Created if it does not already exist.

        Raises:
            IOError: If the manifest file cannot be written.
        """
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "manifest.json")

        entries: list[dict[str, Any]] = sorted(
            [
                {
                    "sequence_id": rec.sequence_id,
                    "risk_class": rec.risk_class,
                    "scenario_name": rec.scenario_name,
                    "num_agents": rec.num_agents,
                    "num_timesteps": rec.num_timesteps,
                    "generation_seed": rec.generation_seed,
                    "label_distribution": dict(rec.label_distribution),
                    "file_path": f"{rec.sequence_id}.json",
                }
                for rec in records
            ],
            key=lambda e: e["sequence_id"],
        )

        try:
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(entries, fh, indent=2, sort_keys=True)
                fh.write("\n")
        except OSError as exc:
            raise IOError(
                f"Failed to write manifest to '{manifest_path}': {exc}"
            ) from exc

        logger.info(
            "Manifest written: %d record(s) → %s", len(records), manifest_path
        )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def generate_dataset(config_path: str) -> None:
    """Orchestrates the full data generation pipeline.

    Reads simulation settings from ``config_path`` (``simulation`` section),
    builds all three scenario configs via ``ScenarioFactory``, runs each
    scenario ``num_runs_per_scenario`` times via ``SimulationRunner``,
    labels each run via ``AutoLabeler``, constructs ``TrajectoryRecord``
    instances, saves them with ``DatasetSerializer``, and writes
    ``manifest.json``.

    Seed strategy: each run receives a globally unique seed computed as::

        seed = random_seed_base + (scenario_index * num_runs_per_scenario) + run_index

    This guarantees that every run—across all scenarios—has a distinct seed
    while remaining fully reproducible from ``random_seed_base`` alone.

    Prints a summary table to stdout on completion showing total runs,
    per-class label distribution, and total dataset size on disk.

    Args:
        config_path: Path to the YAML configuration file (typically
            ``configs/default.yaml``). Must contain a ``simulation`` section
            with keys: ``num_runs_per_scenario``, ``num_timesteps``,
            ``random_seed_base``, ``output_dir``.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        KeyError: If the ``simulation`` section or a required key is missing.
    """
    t_start = time.monotonic()

    sim_cfg = _load_simulation_config(config_path)
    num_runs: int = sim_cfg["num_runs_per_scenario"]
    num_timesteps: int = sim_cfg["num_timesteps"]
    seed_base: int = sim_cfg["random_seed_base"]
    output_dir: str = sim_cfg["output_dir"]

    logger.info(
        "Dataset generation started — %d scenarios × %d runs × %d timesteps → %s",
        3, num_runs, num_timesteps, output_dir,
    )

    scenarios = ScenarioFactory.build_all()
    runner = SimulationRunner()
    labeler = AutoLabeler()
    serializer = DatasetSerializer()
    records: list[TrajectoryRecord] = []

    for scenario_idx, config in enumerate(scenarios):
        logger.info(
            "Scenario %d/3: %s (%s)",
            scenario_idx + 1, config.scenario_name, config.risk_class,
        )
        for run_idx in range(num_runs):
            seed = seed_base + scenario_idx * num_runs + run_idx
            sequence_id = f"{config.scenario_name}_run_{run_idx:03d}"

            positions, velocities = runner.run(config, seed=seed, num_timesteps=num_timesteps)

            frame_labels = labeler.label_sequence(positions, velocities, config)
            label_distribution = AutoLabeler.compute_label_distribution(frame_labels)

            record = TrajectoryRecord(
                sequence_id=sequence_id,
                risk_class=config.risk_class,
                scenario_name=config.scenario_name,
                num_agents=config.num_agents,
                num_timesteps=num_timesteps,
                generation_seed=seed,
                label_distribution=label_distribution,
                positions=positions.tolist(),
                velocities=velocities.tolist(),
                frame_labels=frame_labels,
            )

            serializer.save_record(record, output_dir)
            records.append(record)

            if (run_idx + 1) % 10 == 0 or run_idx == num_runs - 1:
                logger.info(
                    "  %s: %d/%d runs complete", config.scenario_name, run_idx + 1, num_runs
                )

    serializer.write_manifest(records, output_dir)

    elapsed = time.monotonic() - t_start
    _print_summary(records, elapsed, output_dir)


def _load_simulation_config(config_path: str) -> dict[str, Any]:
    """Loads and returns the ``simulation`` section from a YAML config file.

    Validates that all required keys are present so that callers receive
    a ``KeyError`` with a clear message rather than a silent ``None`` when
    a key is missing.

    Required keys: ``num_runs_per_scenario``, ``num_timesteps``,
    ``random_seed_base``, ``output_dir``.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing simulation configuration keys.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If the YAML file has no ``simulation`` section, or if
            a required key is absent from that section.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: '{config_path}'"
        )

    with open(config_path, encoding="utf-8") as fh:
        full_config: dict[str, Any] = yaml.safe_load(fh)

    if "simulation" not in full_config:
        raise KeyError(
            f"YAML file '{config_path}' has no 'simulation' section. "
            "Add a 'simulation:' block with keys: num_runs_per_scenario, "
            "num_timesteps, random_seed_base, output_dir."
        )

    sim: dict[str, Any] = full_config["simulation"]
    required_keys = ("num_runs_per_scenario", "num_timesteps", "random_seed_base", "output_dir")
    missing = [k for k in required_keys if k not in sim]
    if missing:
        raise KeyError(
            f"simulation section in '{config_path}' is missing required key(s): {missing}"
        )

    return sim


def _print_summary(
    records: list[TrajectoryRecord],
    elapsed_seconds: float,
    output_dir: str,
) -> None:
    """Prints a formatted dataset generation summary to stdout.

    Displays total run count, per-scenario breakdown, aggregate label
    distribution across all timesteps, and elapsed wall-clock time.

    Args:
        records: All TrajectoryRecord instances produced in this run.
        elapsed_seconds: Total wall-clock time for generation in seconds.
        output_dir: Directory where files were written.
    """
    total_timesteps = sum(r.num_timesteps for r in records)
    aggregate: dict[str, int] = {cls: 0 for cls in RISK_CLASSES}
    for rec in records:
        for cls, count in rec.label_distribution.items():
            aggregate[cls] += count

    # Group by scenario for the per-scenario row
    scenario_counts: dict[str, int] = {}
    for rec in records:
        scenario_counts[rec.scenario_name] = scenario_counts.get(rec.scenario_name, 0) + 1

    sep = "─" * 52
    print(f"\n{'CrowdFlow DNA — Dataset Generation Complete':^52}")
    print(sep)
    print(f"  Output directory : {output_dir}")
    print(f"  Total sequences  : {len(records)}")
    print(f"  Total timesteps  : {total_timesteps:,}")
    print(f"  Elapsed time     : {elapsed_seconds:.1f}s")
    print(sep)
    print("  Per-scenario sequences:")
    for name, count in sorted(scenario_counts.items()):
        print(f"    {name:<30} {count:>4} runs")
    print(sep)
    print("  Aggregate label distribution (timesteps):")
    for cls in RISK_CLASSES:
        pct = aggregate[cls] / total_timesteps * 100 if total_timesteps else 0.0
        print(f"    {cls:<12} {aggregate[cls]:>8,}  ({pct:5.1f}%)")
    print(sep)



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CrowdFlow DNA — Synthetic crowd data generation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file (default: configs/default.yaml).",
    )
    args = parser.parse_args()
    generate_dataset(config_path=args.config)
