"""
Smoke tests for AutoLabeler in training/simulate_data.py.

Tests verify:
- label_sequence() returns the correct length and valid labels.
- _apply_rules() returns "Safe" for low-metric crowd states.
- _apply_rules() returns "Congesting" when density or divergence passes
  the Congesting threshold.
- _apply_rules() returns "Critical" via the density-driven path (no panic).
- _apply_rules() returns "Critical" via the panic path (enable_panic=True).
- compute_label_distribution() always includes all three risk classes.
- _compute_metrics() edge cases: single agent, all agents stationary.
"""

import numpy as np
import pytest

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from training.simulate_data import (
    AutoLabeler,
    ScenarioConfig,
    DENSITY_CONGESTING_THRESH,
    DENSITY_CRITICAL_THRESH,
    DIVERGENCE_CONGESTING_THRESH,
    DIVERGENCE_CRITICAL_THRESH,
    SPEED_CRITICAL_THRESH,
    RISK_CLASSES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(enable_panic: bool = False) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_name="test_scenario",
        risk_class="Safe",
        num_agents=5,
        scene_width=20.0,
        scene_height=20.0,
        initial_speed=1.0,
        goal_spread=5.0,
        enable_panic=enable_panic,
        description="Test scenario",
    )


def _metrics(density: float, speed: float, divergence: float) -> dict[str, float]:
    return {
        "mean_local_density": density,
        "mean_speed": speed,
        "velocity_divergence": divergence,
    }


# ---------------------------------------------------------------------------
# label_sequence
# ---------------------------------------------------------------------------

class TestLabelSequence:
    def test_output_length_matches_timesteps(self):
        labeler = AutoLabeler()
        T, N = 10, 5
        pos = np.zeros((T, N, 2), dtype=np.float32)
        vel = np.zeros((T, N, 2), dtype=np.float32)
        labels = labeler.label_sequence(pos, vel, _make_config())
        assert len(labels) == T

    def test_all_labels_are_valid_risk_classes(self):
        labeler = AutoLabeler()
        T, N = 20, 4
        pos = np.random.default_rng(0).random((T, N, 2)).astype(np.float32) * 10
        vel = np.random.default_rng(1).random((T, N, 2)).astype(np.float32)
        labels = labeler.label_sequence(pos, vel, _make_config())
        assert all(lbl in RISK_CLASSES for lbl in labels)

    def test_deterministic_on_identical_inputs(self):
        labeler = AutoLabeler()
        T, N = 5, 3
        pos = np.ones((T, N, 2), dtype=np.float32)
        vel = np.ones((T, N, 2), dtype=np.float32) * 0.5
        config = _make_config()
        assert labeler.label_sequence(pos, vel, config) == labeler.label_sequence(pos, vel, config)


# ---------------------------------------------------------------------------
# _apply_rules
# ---------------------------------------------------------------------------

class TestApplyRules:
    def test_safe_when_all_metrics_low(self):
        labeler = AutoLabeler()
        m = _metrics(density=0.0, speed=0.5, divergence=0.0)
        assert labeler._apply_rules(m, enable_panic=False) == "Safe"

    def test_congesting_via_high_density(self):
        labeler = AutoLabeler()
        m = _metrics(density=DENSITY_CONGESTING_THRESH + 0.05, speed=0.5, divergence=0.0)
        assert labeler._apply_rules(m, enable_panic=False) == "Congesting"

    def test_congesting_via_high_divergence(self):
        labeler = AutoLabeler()
        m = _metrics(density=0.0, speed=0.5, divergence=DIVERGENCE_CONGESTING_THRESH + 0.05)
        assert labeler._apply_rules(m, enable_panic=False) == "Congesting"

    def test_critical_via_density_path(self):
        labeler = AutoLabeler()
        m = _metrics(
            density=DENSITY_CRITICAL_THRESH + 0.05,
            speed=0.5,
            divergence=DIVERGENCE_CONGESTING_THRESH + 0.05,
        )
        assert labeler._apply_rules(m, enable_panic=False) == "Critical"

    def test_critical_via_panic_divergence(self):
        labeler = AutoLabeler()
        m = _metrics(density=0.0, speed=0.5, divergence=DIVERGENCE_CRITICAL_THRESH + 0.05)
        assert labeler._apply_rules(m, enable_panic=True) == "Critical"

    def test_critical_via_panic_speed(self):
        labeler = AutoLabeler()
        m = _metrics(density=0.0, speed=SPEED_CRITICAL_THRESH + 0.1, divergence=0.0)
        assert labeler._apply_rules(m, enable_panic=True) == "Critical"

    def test_panic_flag_does_not_trigger_critical_if_metrics_low(self):
        """enable_panic alone does not classify Critical without threshold breach."""
        labeler = AutoLabeler()
        m = _metrics(density=0.0, speed=0.5, divergence=0.0)
        assert labeler._apply_rules(m, enable_panic=True) == "Safe"

    def test_critical_density_path_requires_both_thresholds(self):
        """High density alone (without divergence) should NOT trigger Critical."""
        labeler = AutoLabeler()
        m = _metrics(
            density=DENSITY_CRITICAL_THRESH + 0.05,
            speed=0.5,
            divergence=0.0,   # below DIVERGENCE_CONGESTING_THRESH
        )
        # Should be Congesting (density path), not Critical
        result = labeler._apply_rules(m, enable_panic=False)
        assert result == "Congesting"


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_single_agent_density_is_zero(self):
        labeler = AutoLabeler()
        pos = np.array([[5.0, 5.0]])
        vel = np.array([[1.0, 0.0]])
        m = labeler._compute_metrics(pos, vel)
        assert m["mean_local_density"] == 0.0

    def test_all_stationary_divergence_is_zero(self):
        labeler = AutoLabeler()
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        vel = np.zeros((3, 2))
        m = labeler._compute_metrics(pos, vel)
        assert m["velocity_divergence"] == 0.0
        assert m["mean_speed"] == 0.0

    def test_aligned_agents_divergence_near_zero(self):
        labeler = AutoLabeler()
        pos = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        vel = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        m = labeler._compute_metrics(pos, vel)
        assert m["velocity_divergence"] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_agents_divergence_near_one(self):
        labeler = AutoLabeler()
        pos = np.array([[0.0, 0.0], [5.0, 0.0]])
        vel = np.array([[1.0, 0.0], [-1.0, 0.0]])
        m = labeler._compute_metrics(pos, vel)
        # cos_sim = -1, dissimilarity = 1 - (-1) = 2.0
        assert m["velocity_divergence"] == pytest.approx(2.0, abs=1e-6)

    def test_density_normalized_in_unit_range(self):
        labeler = AutoLabeler()
        # 4 agents all within 2m of each other
        pos = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        vel = np.ones((4, 2))
        m = labeler._compute_metrics(pos, vel)
        assert 0.0 <= m["mean_local_density"] <= 1.0


# ---------------------------------------------------------------------------
# compute_label_distribution
# ---------------------------------------------------------------------------

class TestComputeLabelDistribution:
    def test_all_keys_always_present(self):
        dist = AutoLabeler.compute_label_distribution(["Safe", "Safe"])
        assert set(dist.keys()) == set(RISK_CLASSES)

    def test_correct_counts(self):
        labels = ["Safe", "Safe", "Congesting", "Critical", "Safe"]
        dist = AutoLabeler.compute_label_distribution(labels)
        assert dist["Safe"] == 3
        assert dist["Congesting"] == 1
        assert dist["Critical"] == 1

    def test_empty_list_all_zeros(self):
        dist = AutoLabeler.compute_label_distribution([])
        assert all(v == 0 for v in dist.values())

    def test_distribution_total_matches_label_count(self):
        labels = ["Safe"] * 50 + ["Critical"] * 10
        dist = AutoLabeler.compute_label_distribution(labels)
        assert sum(dist.values()) == 60
