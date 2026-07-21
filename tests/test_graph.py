"""
Tests for crowdflow_dna/graph/graph_builder.py.

Strategy: _build_node_features(), _build_edges(), and _validate_inputs() are
tested directly with numpy/torch only so that the entire test class runs in
the lightweight CI environment without requiring torch_geometric.

The build() method (which requires torch_geometric) is guarded by a skipif
decorator and skipped gracefully when the package is absent.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from crowdflow_dna.graph.graph_builder import GraphBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _builder(radius: float = 0.5) -> GraphBuilder:
    return GraphBuilder(proximity_radius=radius)


def _pos(*rows: tuple[float, float]) -> np.ndarray:
    """Shorthand: create a float32 position array of shape (N, 2)."""
    return np.array(rows, dtype=np.float32)


def _vel(*rows: tuple[float, float]) -> np.ndarray:
    """Shorthand: create a float32 velocity array of shape (N, 2)."""
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# GraphBuilder construction
# ---------------------------------------------------------------------------

class TestGraphBuilderInit:
    def test_valid_radius_stored(self):
        gb = GraphBuilder(proximity_radius=0.2)
        assert gb.proximity_radius == 0.2

    def test_radius_of_one_accepted(self):
        gb = GraphBuilder(proximity_radius=1.0)
        assert gb.proximity_radius == 1.0

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError, match="proximity_radius"):
            GraphBuilder(proximity_radius=0.0)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError):
            GraphBuilder(proximity_radius=-0.1)

    def test_above_one_accepted(self):
        gb = GraphBuilder(proximity_radius=2.0)
        assert gb.proximity_radius == 2.0


# ---------------------------------------------------------------------------
# _validate_inputs
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_valid_inputs_pass_silently(self):
        GraphBuilder._validate_inputs(_pos((0.1, 0.2)), _vel((0.3, 0.4)))

    def test_positions_1d_raises(self):
        with pytest.raises(ValueError, match="positions"):
            GraphBuilder._validate_inputs(
                np.array([0.1, 0.2], dtype=np.float32), _vel((0.0, 0.0))
            )

    def test_positions_3_columns_raises(self):
        with pytest.raises(ValueError, match="positions"):
            GraphBuilder._validate_inputs(
                np.zeros((2, 3), dtype=np.float32), _vel((0.0, 0.0), (0.0, 0.0))
            )

    def test_velocities_1d_raises(self):
        with pytest.raises(ValueError, match="velocities"):
            GraphBuilder._validate_inputs(
                _pos((0.1, 0.2)), np.array([0.3, 0.4], dtype=np.float32)
            )

    def test_mismatched_agent_count_raises(self):
        with pytest.raises(ValueError, match="same number of agents"):
            GraphBuilder._validate_inputs(
                _pos((0.1, 0.2), (0.3, 0.4)), _vel((0.0, 0.0))
            )



# ---------------------------------------------------------------------------
# _compute_speed
# ---------------------------------------------------------------------------

class TestComputeSpeed:
    def test_empty_array_returns_empty(self):
        result = GraphBuilder._compute_speed(np.zeros((0, 2), dtype=np.float32))
        assert result.shape == (0,)
        assert result.dtype == np.float32

    def test_known_3_4_5_right_triangle(self):
        vel = _vel((3.0, 4.0))
        result = GraphBuilder._compute_speed(vel)
        assert result[0] == pytest.approx(5.0, abs=1e-6)

    def test_stationary_agent_speed_is_zero(self):
        result = GraphBuilder._compute_speed(_vel((0.0, 0.0)))
        assert result[0] == pytest.approx(0.0)

    def test_output_shape_is_n(self):
        vel = _vel((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        result = GraphBuilder._compute_speed(vel)
        assert result.shape == (3,)

    def test_output_dtype_is_float32(self):
        result = GraphBuilder._compute_speed(_vel((1.0, 2.0)))
        assert result.dtype == np.float32

    def test_unit_vector_speed_is_one(self):
        result = GraphBuilder._compute_speed(_vel((1.0, 0.0)))
        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_multiple_agents_correct_speeds(self):
        vel = _vel((3.0, 4.0), (0.0, 0.0), (1.0, 0.0))
        result = GraphBuilder._compute_speed(vel)
        assert result[0] == pytest.approx(5.0, abs=1e-6)
        assert result[1] == pytest.approx(0.0, abs=1e-6)
        assert result[2] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _build_node_features
# ---------------------------------------------------------------------------

class TestBuildNodeFeatures:
    def test_empty_returns_zero_tensor(self):
        feats = GraphBuilder._build_node_features(
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
        assert feats.shape == (0, 5)
        assert feats.dtype == torch.float32

    def test_shape_is_n_by_5(self):
        pos = _pos((0.1, 0.2), (0.3, 0.4), (0.5, 0.6))
        vel = _vel((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        feats = GraphBuilder._build_node_features(pos, vel)
        assert feats.shape == (3, 5)

    def test_x_y_columns_match_positions(self):
        pos = _pos((0.1, 0.9))
        vel = _vel((0.0, 0.0))
        feats = GraphBuilder._build_node_features(pos, vel)
        assert torch.allclose(feats[0, :2], torch.tensor([0.1, 0.9]))

    def test_vx_vy_columns_match_velocities(self):
        pos = _pos((0.5, 0.5))
        vel = _vel((0.3, 0.7))
        feats = GraphBuilder._build_node_features(pos, vel)
        assert torch.allclose(feats[0, 2:4], torch.tensor([0.3, 0.7]))

    def test_speed_column_is_l2_norm(self):
        pos = _pos((0.0, 0.0))
        vel = _vel((3.0, 4.0))  # speed = 5.0
        feats = GraphBuilder._build_node_features(pos, vel)
        assert torch.allclose(feats[0, 4], torch.tensor(5.0), atol=1e-5)

    def test_stationary_agent_has_zero_speed(self):
        pos = _pos((0.5, 0.5))
        vel = _vel((0.0, 0.0))
        feats = GraphBuilder._build_node_features(pos, vel)
        assert feats[0, 4].item() == pytest.approx(0.0)

    def test_dtype_is_float32(self):
        feats = GraphBuilder._build_node_features(_pos((0.1, 0.2)), _vel((0.1, 0.2)))
        assert feats.dtype == torch.float32

    def test_order_of_agents_preserved(self):
        pos = _pos((0.1, 0.2), (0.8, 0.9))
        vel = _vel((1.0, 0.0), (0.0, 2.0))
        feats = GraphBuilder._build_node_features(pos, vel)
        assert torch.allclose(feats[0, :2], torch.tensor([0.1, 0.2]))
        assert torch.allclose(feats[1, :2], torch.tensor([0.8, 0.9]))


# ---------------------------------------------------------------------------
# _build_edges
# ---------------------------------------------------------------------------

class TestBuildEdges:
    def test_empty_positions_no_edges(self):
        gb = _builder()
        ei, ea = gb._build_edges(
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
        assert ei.shape == (2, 0)
        assert ea.shape == (0, 4)

    def test_single_agent_no_edges(self):
        gb = _builder()
        ei, ea = gb._build_edges(_pos((0.5, 0.5)), _vel((0.0, 0.0)))
        assert ei.shape == (2, 0)
        assert ea.shape == (0, 4)

    def test_two_nearby_agents_produce_two_directed_edges(self):
        gb = _builder(radius=0.2)
        ei, ea = gb._build_edges(
            _pos((0.0, 0.0), (0.1, 0.0)),
            _vel((0.0, 0.0), (0.0, 0.0)),
        )
        assert ei.shape[1] == 2

    def test_two_far_agents_produce_no_edges(self):
        gb = _builder(radius=0.05)
        ei, ea = gb._build_edges(
            _pos((0.0, 0.0), (0.5, 0.0)),
            _vel((0.0, 0.0), (0.0, 0.0)),
        )
        assert ei.shape[1] == 0

    def test_edge_exactly_at_radius_is_included(self):
        gb = _builder(radius=0.1)
        ei, ea = gb._build_edges(
            _pos((0.0, 0.0), (0.1, 0.0)),
            _vel((0.0, 0.0), (0.0, 0.0)),
        )
        assert ei.shape[1] == 2

    def test_edge_attr_shape_is_e_by_4(self):
        gb = _builder(radius=0.5)
        ei, ea = gb._build_edges(
            _pos((0.0, 0.0), (0.1, 0.0)),
            _vel((1.0, 0.0), (0.0, 1.0)),
        )
        assert ea.shape == (ei.shape[1], 4)

    def test_distance_column_is_correct(self):
        gb = _builder(radius=1.0)
        # distance = 0.5
        ei, ea = gb._build_edges(
            _pos((0.0, 0.0), (0.3, 0.4)),
            _vel((0.0, 0.0), (0.0, 0.0)),
        )
        assert ei.shape[1] == 2
        # column index 2 is distance
        assert torch.allclose(ea[:, 2], torch.tensor([0.5, 0.5]), atol=1e-5)

    def test_dx_dy_are_antisymmetric(self):
        """Forward edge (dx, dy) must be the negative of the reverse edge."""
        gb = _builder(radius=1.0)
        pos = _pos((0.1, 0.2), (0.4, 0.6))
        vel = _vel((0.0, 0.0), (0.0, 0.0))
        ei, ea = gb._build_edges(pos, vel)
        assert ei.shape[1] == 2
        # Find which edge is forward (src < dst) and which is reverse
        src, dst = ei[0].tolist(), ei[1].tolist()
        fwd_idx = 0 if src[0] < dst[0] else 1
        rev_idx = 1 - fwd_idx
        assert torch.allclose(ea[fwd_idx, :2], -ea[rev_idx, :2], atol=1e-5)

    def test_relative_speed_is_symmetric(self):
        gb = _builder(radius=1.0)
        ei, ea = gb._build_edges(
            _pos((0.0, 0.0), (0.1, 0.0)),
            _vel((1.0, 0.0), (0.0, 1.0)),
        )
        # Column 3 is relative_speed — must be the same for both directions
        assert torch.allclose(ea[0, 3], ea[1, 3], atol=1e-5)

    def test_stationary_agents_have_zero_relative_speed(self):
        gb = _builder(radius=1.0)
        ei, ea = gb._build_edges(
            _pos((0.0, 0.0), (0.1, 0.0)),
            _vel((0.5, 0.5), (0.5, 0.5)),  # identical velocities
        )
        assert ea[:, 3].abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_no_self_loops(self):
        gb = _builder(radius=1.0)
        ei, _ = gb._build_edges(
            _pos((0.1, 0.2), (0.3, 0.4)),
            _vel((0.0, 0.0), (0.0, 0.0)),
        )
        src, dst = ei[0], ei[1]
        assert not (src == dst).any()

    def test_graph_is_symmetric(self):
        gb = _builder(radius=1.0)
        pos = _pos((0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0))
        vel = _vel((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        ei, _ = gb._build_edges(pos, vel)
        edges = set(zip(ei[0].tolist(), ei[1].tolist()))
        for s, d in list(edges):
            assert (d, s) in edges

    def test_three_agents_all_close_six_directed_edges(self):
        gb = _builder(radius=0.15)
        pos = _pos((0.0, 0.0), (0.1, 0.0), (0.05, 0.087))
        vel = _vel((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        ei, _ = gb._build_edges(pos, vel)
        assert ei.shape[1] == 6

    def test_edge_index_dtype_is_long(self):
        gb = _builder(radius=1.0)
        ei, _ = gb._build_edges(_pos((0.0, 0.0), (0.1, 0.0)), _vel((0.0, 0.0), (0.0, 0.0)))
        assert ei.dtype == torch.long

    def test_edge_attr_dtype_is_float32(self):
        gb = _builder(radius=1.0)
        _, ea = gb._build_edges(_pos((0.0, 0.0), (0.1, 0.0)), _vel((0.0, 0.0), (0.0, 0.0)))
        assert ea.dtype == torch.float32


# ---------------------------------------------------------------------------
# build() — requires torch_geometric, skipped if unavailable
# ---------------------------------------------------------------------------


def _can_import_torch_geometric() -> bool:
    try:
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False


torch_geometric_available = pytest.mark.skipif(
    not _can_import_torch_geometric(),
    reason="torch_geometric not installed",
)


@torch_geometric_available
class TestBuild:
    def test_empty_returns_valid_data_object(self):
        from torch_geometric.data import Data
        gb = _builder()
        g = gb.build(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))
        assert isinstance(g, Data)
        assert g.num_nodes == 0
        assert g.edge_index.shape == (2, 0)

    def test_single_agent_no_edges(self):
        gb = _builder()
        g = gb.build(_pos((0.5, 0.5)), _vel((0.0, 0.0)))
        assert g.num_nodes == 1
        assert g.edge_index.shape[1] == 0

    def test_two_nearby_agents_graph(self):
        gb = _builder(radius=0.2)
        g = gb.build(_pos((0.0, 0.0), (0.1, 0.0)), _vel((1.0, 0.0), (0.0, 1.0)))
        assert g.num_nodes == 2
        assert g.edge_index.shape == (2, 2)
        assert g.x.shape == (2, 5)
        assert g.edge_attr.shape == (2, 4)

    def test_node_feature_shape_is_n_by_5(self):
        gb = _builder(radius=1.0)
        pos = np.random.default_rng(0).random((7, 2)).astype(np.float32)
        vel = np.random.default_rng(1).random((7, 2)).astype(np.float32)
        g = gb.build(pos, vel)
        assert g.x.shape == (7, 5)

    def test_mismatched_shapes_raise_value_error(self):
        gb = _builder()
        with pytest.raises(ValueError, match="same number of agents"):
            gb.build(_pos((0.1, 0.2), (0.3, 0.4)), _vel((0.0, 0.0)))

    def test_deterministic_output(self):
        gb = _builder(radius=0.5)
        pos = _pos((0.1, 0.2), (0.3, 0.4))
        vel = _vel((0.5, 0.6), (0.7, 0.8))
        g1 = gb.build(pos, vel)
        g2 = gb.build(pos, vel)
        assert torch.equal(g1.x, g2.x)
        assert torch.equal(g1.edge_index, g2.edge_index)
        assert torch.equal(g1.edge_attr, g2.edge_attr)

    def test_build_import_error_without_pyg(self, monkeypatch):
        """build() should raise ImportError with a helpful message if PyG missing."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch_geometric.data":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        gb = _builder()
        with pytest.raises(ImportError, match="torch_geometric"):
            gb.build(_pos((0.5, 0.5)), _vel((0.0, 0.0)))
