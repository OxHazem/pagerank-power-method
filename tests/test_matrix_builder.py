"""
Unit tests for matrix_builder.py
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "03. Deliverable 3 - Code Implementation & Documentation"))

from matrix_builder import build_matrix


class TestBuildMatrix:
    """Test the build_matrix function."""

    def test_basic_triangle(self, simple_triangle_edges):
        """Test matrix construction for a simple triangle graph."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        
        # Check dimensions
        assert P.shape == (3, 3)
        assert v.shape == (3,)
        
        # Check that v sums to 1
        assert np.isclose(v.sum(), 1.0)
        
        # Check that v is uniform for no personalization
        assert np.allclose(v, 1/3)

    def test_alpha_validation(self, simple_triangle_edges):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="Damping factor alpha must be in"):
            build_matrix(simple_triangle_edges, alpha=1.5)
        
        with pytest.raises(ValueError, match="Damping factor alpha must be in"):
            build_matrix(simple_triangle_edges, alpha=-0.1)

    def test_empty_edges(self):
        """Test that empty edge list raises ValueError."""
        with pytest.raises(ValueError, match="Edge list is empty"):
            build_matrix([])

    def test_column_stochastic_property(self, simple_triangle_edges):
        """Test that P is column-stochastic (non-dangling columns sum to 1)."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        
        # For non-dangling nodes, columns should sum to 1
        col_sums = P.sum(axis=0).A1  # Convert to dense array
        # All columns should sum to 1
        assert np.allclose(col_sums, 1.0)

    def test_dangling_nodes_handling(self, isolated_node_edges):
        """Test that dangling nodes are handled correctly."""
        # Edges: (0, 1), (1, 2) -> node 2 is dangling
        P, v = build_matrix(isolated_node_edges, alpha=0.85)
        
        n = max(max(e[0], e[1]) for e in isolated_node_edges) + 1
        
        # Check that P is column-stochastic
        col_sums = P.sum(axis=0).A1
        assert np.allclose(col_sums, 1.0)

    def test_weighted_edges(self, weighted_edges):
        """Test matrix construction with weighted edges."""
        P, v = build_matrix(weighted_edges, alpha=0.85)
        
        # Check dimensions
        n = 3
        assert P.shape == (n, n)
        
        # Check column stochasticity
        col_sums = P.sum(axis=0).A1
        assert np.allclose(col_sums, 1.0)

    def test_alpha_boundary_values(self, simple_triangle_edges):
        """Test that alpha=0 and alpha=1 are allowed."""
        P0, v0 = build_matrix(simple_triangle_edges, alpha=0.0)
        P1, v1 = build_matrix(simple_triangle_edges, alpha=1.0)
        
        assert P0.shape == (3, 3)
        assert P1.shape == (3, 3)

    def test_personalization_uniform(self, simple_triangle_edges):
        """Test personalization with a single node."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85, personalization=0)
        
        # v should be concentrated on node 0
        assert np.isclose(v[0], 1.0)
        assert np.isclose(v[1], 0.0)
        assert np.isclose(v[2], 0.0)
        assert np.isclose(v.sum(), 1.0)

    def test_personalization_out_of_bounds(self, simple_triangle_edges):
        """Test that out-of-bounds personalization raises ValueError."""
        with pytest.raises(ValueError, match="out of bounds"):
            build_matrix(simple_triangle_edges, personalization=10)

    def test_self_loop(self, self_loop_edges):
        """Test matrix construction with self-loops."""
        P, v = build_matrix(self_loop_edges, alpha=0.85)
        
        # Check dimensions
        n = 3
        assert P.shape == (n, n)
        
        # Check column stochasticity
        col_sums = P.sum(axis=0).A1
        assert np.allclose(col_sums, 1.0)

    def test_matrix_is_sparse(self, simple_triangle_edges):
        """Test that matrix is returned as sparse."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        
        # Check that P is sparse
        assert hasattr(P, 'toarray')  # csr_matrix has this method

    def test_teleportation_vector_sum(self, simple_triangle_edges):
        """Test that teleportation vector sums to 1."""
        for alpha in [0.0, 0.5, 0.85, 1.0]:
            P, v = build_matrix(simple_triangle_edges, alpha=alpha)
            assert np.isclose(v.sum(), 1.0)

    def test_large_graph_dimensions(self):
        """Test with a larger graph."""
        # Create a larger graph
        edges = [(i, (i+1) % 100, 1.0) for i in range(100)]
        P, v = build_matrix(edges, alpha=0.85)
        
        assert P.shape == (100, 100)
        assert v.shape == (100,)
        assert np.isclose(v.sum(), 1.0)
