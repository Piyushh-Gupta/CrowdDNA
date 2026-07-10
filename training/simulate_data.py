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
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

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
        risk_class: Scenario-level intended class.
        scenario_name: Source scenario (matches ``ScenarioConfig.scenario_name``).
        num_agents: Number of agents tracked in this run.
        num_timesteps: Number of timesteps actually recorded.
        generation_seed: Random seed used; enables exact reproduction.
        label_distribution: Per-class timestep counts across the full run.
        positions: Agent positions per timestep; shape ``(T, N, 2)``,
            stored as nested Python lists for JSON compatibility.
        velocities: Agent velocities per timestep; shape ``(T, N, 2)``,
            stored as nested Python lists for JSON compatibility.
        frame_labels: Per-timestep risk label; length ``T``.
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
    """

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
        pass  # TODO: implement PySocialForce simulation loop

    def _build_initial_state(
        self,
        config: ScenarioConfig,
    ) -> np.ndarray:
        """Samples random initial positions and goal positions for all agents.

        Args:
            config: Active scenario configuration.

        Returns:
            NumPy array of shape ``(N, 6)`` where columns are
            ``[px, py, vx, vy, gx, gy]`` — the PySocialForce initial
            state format.
        """
        pass  # TODO: implement initial state sampling

    def _build_obstacles(
        self,
        config: ScenarioConfig,
    ) -> list[Any] | None:
        """Constructs PySocialForce obstacle definitions from scenario config.

        Args:
            config: Active scenario configuration.

        Returns:
            List of obstacle objects for PySocialForce, or ``None`` if the
            scenario has no obstacles.
        """
        pass  # TODO: implement obstacle construction


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
    """

    def label_sequence(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        config: ScenarioConfig,
    ) -> list[str]:
        """Labels all timesteps in a trajectory sequence.

        Args:
            positions: Agent positions of shape ``(T, N, 2)``.
            velocities: Agent velocities of shape ``(T, N, 2)``.
            config: Scenario config; used to consult ``enable_panic`` flag.

        Returns:
            List of risk label strings of length ``T``, each element being
            one of ``"Safe"``, ``"Congesting"``, or ``"Critical"``.
        """
        pass  # TODO: iterate over timesteps and call _compute_metrics + _apply_rules

    def _compute_metrics(
        self,
        pos_t: np.ndarray,
        vel_t: np.ndarray,
    ) -> dict[str, float]:
        """Computes crowd-level kinematic metrics for a single timestep.

        Args:
            pos_t: Agent positions at timestep t; shape ``(N, 2)``.
            vel_t: Agent velocities at timestep t; shape ``(N, 2)``.

        Returns:
            Dictionary with keys:
                - ``"mean_local_density"`` (float): Normalized average
                  number of neighbours within ``PROXIMITY_RADIUS_M``.
                - ``"mean_speed"`` (float): Mean L2 velocity norm across
                  all agents (m/s).
                - ``"velocity_divergence"`` (float): Mean of
                  ``(1 - cosine_similarity)`` across all agent pairs;
                  0 = perfectly aligned, 1 = fully chaotic.
        """
        pass  # TODO: implement metric computation

    def _apply_rules(
        self,
        metrics: dict[str, float],
        enable_panic: bool,
    ) -> str:
        """Applies priority-ordered labeling rules to a metrics snapshot.

        Priority order (highest to lowest): Critical → Congesting → Safe.

        Args:
            metrics: Output of ``_compute_metrics`` for one timestep.
            enable_panic: Whether the active scenario has panic forces
                enabled. Used to lower the Critical classification bar.

        Returns:
            One of ``"Safe"``, ``"Congesting"``, or ``"Critical"``.
        """
        pass  # TODO: implement threshold rule logic

    @staticmethod
    def compute_label_distribution(labels: list[str]) -> dict[str, int]:
        """Counts occurrences of each risk label in a labeled sequence.

        Args:
            labels: List of risk label strings of length T.

        Returns:
            Dictionary mapping each risk class to its timestep count,
            e.g. ``{"Safe": 280, "Congesting": 15, "Critical": 5}``.
        """
        pass  # TODO: implement distribution counting


# ---------------------------------------------------------------------------
# DatasetSerializer
# ---------------------------------------------------------------------------


class DatasetSerializer:
    """Writes TrajectoryRecord objects to JSON and maintains manifest.json.

    The JSON output of this class is the authoritative integration contract
    between simulate_data.py and prepare_datasets.py. The schema is defined
    by the ``TrajectoryRecord.to_dict()`` method.
    """

    def save_record(
        self,
        record: TrajectoryRecord,
        output_dir: str,
    ) -> str:
        """Serializes one TrajectoryRecord to a JSON file.

        Args:
            record: A validated TrajectoryRecord instance.
            output_dir: Directory path where the JSON file will be written.
                Created if it does not already exist.

        Returns:
            Absolute path of the written JSON file.

        Raises:
            IOError: If the file cannot be written.
        """
        pass  # TODO: implement JSON write logic

    def write_manifest(
        self,
        records: list[TrajectoryRecord],
        output_dir: str,
    ) -> None:
        """Writes the dataset manifest to manifest.json in output_dir.

        The manifest contains one entry per TrajectoryRecord with fields:
        ``sequence_id``, ``risk_class``, ``scenario_name``,
        ``num_timesteps``, ``num_agents``, ``file_path``,
        ``label_distribution``, ``generation_seed``.

        Args:
            records: All TrajectoryRecord instances produced in this run.
            output_dir: Directory where manifest.json will be written.
        """
        pass  # TODO: implement manifest write logic


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

    Prints a summary table to stdout on completion showing total runs,
    per-class label distribution, and total dataset size on disk.

    Args:
        config_path: Path to the YAML configuration file (typically
            ``configs/default.yaml``). Must contain a ``simulation`` section
            with keys: ``num_runs_per_scenario``, ``num_timesteps``,
            ``random_seed_base``, ``output_dir``.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        KeyError: If the ``simulation`` section is missing from the YAML.
    """
    pass  # TODO: implement full orchestration logic


def _load_simulation_config(config_path: str) -> dict[str, Any]:
    """Loads and returns the ``simulation`` section from a YAML config file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing simulation configuration keys.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If the YAML file has no ``simulation`` section.
    """
    pass  # TODO: implement YAML loading


def _print_summary(
    records: list[TrajectoryRecord],
    elapsed_seconds: float,
    output_dir: str,
) -> None:
    """Prints a formatted dataset generation summary to stdout.

    Args:
        records: All TrajectoryRecord instances produced in this run.
        elapsed_seconds: Total wall-clock time for generation in seconds.
        output_dir: Directory where files were written.
    """
    pass  # TODO: implement summary printing


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
