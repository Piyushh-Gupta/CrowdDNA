"""
Tests for crowdflow_dna/graph/graph_builder.py.

Strategy: GraphBuilder._build_node_features() and _build_edges() are tested
directly with numpy/torch only, avoiding a PyTorch Geometric import so that
these tests run in the lightweight CI environment.

The build() method (which requires torch_geometric) is tested only when
the package is available; otherwise the tests are skipped.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from crowdflow_dna.graph.graph_builder import Detection, GraphBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det(track_id: int, x: float, y: float, conf: float = 1.0) -> Detection:
    """Shorthand for creating a Detection."""
    return Detection(track_id=track_id, x=x, y=y, confidence=conf)


def _builder(radius: float = 0.15) -> GraphBuilder:
    return GraphBuilder(proximity_radius=radius)


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

class TestDetection:
    def test_fields_accessible(self):
        d = Detection(track_id=1, x=0.5, y=0.3, confidence=0.9)
        assert d.track_id == 1
        assert d.x == 0.5
        assert d.y == 0.3
        assert d.confidence == 0.9


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

    def test_above_one_raises(self):
        with pytest.raises(ValueError):
            GraphBuilder(proximity_radius=1.01)


# ---------------------------------------------------------------------------
# _build_node_features
# ---------------------------------------------------------------------------

class TestBuildNodeFeatures:
    def test_empty_detections_returns_zero_tensors(self):
        gb = _builder()
        feats, ids = gb._build_node_features([])
        assert feats.shape == (0, 3)
        assert ids.shape == (0,)
        assert feats.dtype == torch.float32
        assert ids.dtype == torch.long

    def test_single_detection_shape(self):
        gb = _builder()
        feats, ids = gb._build_node_features([_det(7, 0.3, 0.4, 0.8)])
        assert feats.shape == (1, 3)
        assert ids.shape == (1,)

    def test_feature_values_correct(self):
        gb = _builder()
        feats, ids = gb._build_node_features([_det(3, 0.1, 0.9, 0.75)])
        assert torch.allclose(feats[0], torch.tensor([0.1, 0.9, 0.75]))
        assert ids[0].item() == 3

    def test_multiple_detections_order_preserved(self):
        gb = _builder()
        dets = [_det(1, 0.2, 0.3, 0.9), _det(2, 0.5, 0.6, 0.7)]
        feats, ids = gb._build_node_features(dets)
        assert feats.shape == (2, 3)
        assert ids[0].item() == 1
        assert ids[1].item() == 2

    def test_dtype_is_float32(self):
        gb = _builder()
        feats, _ = gb._build_node_features([_det(0, 0.5, 0.5, 1.0)])
        assert feats.dtype == torch.float32


# ---------------------------------------------------------------------------
# _build_edges
# ---------------------------------------------------------------------------

class TestBuildEdges:
    def test_empty_detections_no_edges(self):
        gb = _builder()
        ei, ea = gb._build_edges([])
        assert ei.shape == (2, 0)
        assert ea.shape == (0, 1)

    def test_single_detection_no_edges(self):
        gb = _builder()
        ei, ea = gb._build_edges([_det(0, 0.5, 0.5)])
        assert ei.shape == (2, 0)
        assert ea.shape == (0, 1)

    def test_two_nearby_nodes_produce_two_directed_edges(self):
        # Distance = 0.1 < radius 0.15 → one undirected edge = 2 directed
        gb = _builder(radius=0.15)
        dets = [_det(0, 0.0, 0.0), _det(1, 0.1, 0.0)]
        ei, ea = gb._build_edges(dets)
        assert ei.shape[1] == 2, "Undirected edge should appear in both directions"
        assert ea.shape == (2, 1)

    def test_two_far_nodes_produce_no_edges(self):
        # Distance = 0.5 > radius 0.15 → no edges
        gb = _builder(radius=0.15)
        dets = [_det(0, 0.0, 0.0), _det(1, 0.5, 0.0)]
        ei, ea = gb._build_edges(dets)
        assert ei.shape[1] == 0

    def test_edge_exactly_at_radius_is_included(self):
        gb = _builder(radius=0.1)
        dets = [_det(0, 0.0, 0.0), _det(1, 0.1, 0.0)]  # distance == 0.1
        ei, ea = gb._build_edges(dets)
        assert ei.shape[1] == 2

    def test_edge_attr_values_are_distances(self):
        gb = _builder(radius=0.5)
        dets = [_det(0, 0.0, 0.0), _det(1, 0.3, 0.4)]  # distance = 0.5
        ei, ea = gb._build_edges(dets)
        assert ei.shape[1] == 2
        assert torch.allclose(ea, torch.tensor([[0.5], [0.5]]), atol=1e-5)

    def test_edge_index_dtype_is_long(self):
        gb = _builder(radius=0.5)
        dets = [_det(0, 0.0, 0.0), _det(1, 0.1, 0.0)]
        ei, _ = gb._build_edges(dets)
        assert ei.dtype == torch.long

    def test_edge_attr_dtype_is_float32(self):
        gb = _builder(radius=0.5)
        dets = [_det(0, 0.0, 0.0), _det(1, 0.1, 0.0)]
        _, ea = gb._build_edges(dets)
        assert ea.dtype == torch.float32

    def test_three_nodes_all_close_six_directed_edges(self):
        # Equilateral triangle with side ~0.1 — all within radius 0.15
        gb = _builder(radius=0.15)
        dets = [
            _det(0, 0.0, 0.0),
            _det(1, 0.1, 0.0),
            _det(2, 0.05, 0.087),  # approximately equilateral
        ]
        ei, ea = gb._build_edges(dets)
        # 3 unique pairs × 2 directions = 6
        assert ei.shape[1] == 6

    def test_no_self_loops(self):
        gb = _builder(radius=1.0)
        dets = [_det(0, 0.5, 0.5), _det(1, 0.6, 0.6)]
        ei, _ = gb._build_edges(dets)
        src, dst = ei[0], ei[1]
        assert not (src == dst).any(), "Self-loops must not be present"

    def test_graph_is_symmetric(self):
        gb = _builder(radius=1.0)
        dets = [_det(i, float(i) * 0.1, 0.0) for i in range(4)]
        ei, _ = gb._build_edges(dets)
        edges = set(zip(ei[0].tolist(), ei[1].tolist()))
        for s, d in list(edges):
            assert (d, s) in edges, "Every edge must have its reverse"


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
        g = gb.build([])
        assert isinstance(g, Data)
        assert g.num_nodes == 0
        assert g.edge_index.shape == (2, 0)

    def test_single_node_no_edges(self):
        gb = _builder()
        g = gb.build([_det(1, 0.5, 0.5)])
        assert g.num_nodes == 1
        assert g.edge_index.shape[1] == 0

    def test_two_nearby_nodes_graph(self):
        gb = _builder(radius=0.15)
        g = gb.build([_det(0, 0.0, 0.0), _det(1, 0.1, 0.0)])
        assert g.num_nodes == 2
        assert g.edge_index.shape == (2, 2)
        assert g.x.shape == (2, 3)

    def test_track_ids_in_data(self):
        gb = _builder(radius=0.5)
        dets = [_det(42, 0.1, 0.1), _det(99, 0.2, 0.2)]
        g = gb.build(dets)
        ids = g.track_ids.tolist()
        assert 42 in ids
        assert 99 in ids

    def test_node_feature_shape(self):
        gb = _builder(radius=0.5)
        dets = [_det(i, float(i) * 0.1, 0.0) for i in range(5)]
        g = gb.build(dets)
        assert g.x.shape == (5, 3)

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
            gb.build([_det(0, 0.5, 0.5)])
