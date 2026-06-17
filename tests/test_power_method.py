"""
Unit tests for power_method.py
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "03. Deliverable 3 - Code Implementation & Documentation"))

from matrix_builder import build_matrix
from power_method import compute_pagerank


class TestComputePageRank:
    """Test the compute_pagerank function."""

    def test_basic_triangle(self, simple_triangle_edges):
        """Test PageRank computation for a simple triangle graph."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        r, residuals, iterations, runtime = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        # Check output dimensions
        assert r.shape == (3,)
        assert len(residuals) > 0
        assert iterations > 0
        assert runtime > 0

    def test_pagerank_sums_to_one(self, simple_triangle_edges):
        """Test that PageRank vector sums to 1."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        r, _, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        assert np.isclose(r.sum(), 1.0)

    def test_pagerank_non_negative(self, simple_triangle_edges):
        """Test that all PageRank scores are non-negative."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        r, _, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        assert np.all(r >= 0)

    def test_convergence_property(self, simple_triangle_edges):
        """Test that residuals generally decrease over iterations."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        r, residuals, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        # First residual should be larger than last (in general)
        # Allow some exceptions due to normalization
        assert residuals[0] > residuals[-1] or len(residuals) == 1

    def test_tolerance_parameter(self, simple_triangle_edges):
        """Test that convergence is controlled by tolerance."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        
        # Loose tolerance
        r_loose, residuals_loose, iter_loose, _ = compute_pagerank(
            P, v, alpha=0.85, tol=1e-1, max_iter=100
        )
        
        # Tight tolerance
        r_tight, residuals_tight, iter_tight, _ = compute_pagerank(
            P, v, alpha=0.85, tol=1e-8, max_iter=100
        )
        
        # Tight tolerance should require more iterations (usually)
        # or achieve better final residual
        assert len(residuals_tight) >= len(residuals_loose) or residuals_tight[-1] <= residuals_loose[-1]

    def test_max_iter_limit(self, simple_triangle_edges):
        """Test that max_iter is respected."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        r, residuals, iterations, _ = compute_pagerank(
            P, v, alpha=0.85, tol=1e-10, max_iter=5
        )
        
        assert iterations <= 5

    def test_dimension_mismatch_P(self):
        """Test that non-square P raises ValueError."""
        P_nonsquare = np.random.rand(3, 4)
        from scipy.sparse import csr_matrix
        P = csr_matrix(P_nonsquare)
        v = np.ones(3) / 3
        
        with pytest.raises(ValueError, match="must be square"):
            compute_pagerank(P, v)

    def test_dimension_mismatch_v(self, simple_triangle_edges):
        """Test that v dimension mismatch raises ValueError."""
        P, _ = build_matrix(simple_triangle_edges, alpha=0.85)
        v_wrong = np.ones(4) / 4
        
        with pytest.raises(ValueError, match="does not match"):
            compute_pagerank(P, v_wrong)

    def test_uniform_graph(self):
        """Test PageRank on a fully connected graph (all nodes equal)."""
        edges = [(0, 1, 1.0), (0, 2, 1.0), (1, 0, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 1, 1.0)]
        P, v = build_matrix(edges, alpha=0.85)
        r, _, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        # All scores should be approximately equal in a symmetric graph
        assert np.allclose(r, 1/3, atol=0.01)

    def test_alpha_zero(self, simple_triangle_edges):
        """Test PageRank with alpha=0 (uniform distribution)."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.0)
        r, _, _, _ = compute_pagerank(P, v, alpha=0.0, tol=1e-6, max_iter=100)
        
        # With alpha=0, result should be uniform teleportation vector
        assert np.allclose(r, v)

    def test_alpha_one(self, simple_triangle_edges):
        """Test PageRank with alpha=1 (pure random walk)."""
        P, v = build_matrix(simple_triangle_edges, alpha=1.0)
        r, _, _, _ = compute_pagerank(P, v, alpha=1.0, tol=1e-6, max_iter=100)
        
        assert np.isclose(r.sum(), 1.0)
        assert np.all(r >= 0)

    def test_runtime_positive(self, simple_triangle_edges):
        """Test that runtime is positive."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        _, _, _, runtime = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        assert runtime >= 0

    def test_residual_monotonicity(self, simple_triangle_edges):
        """Test that residuals follow convergence pattern."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        r, residuals, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-8, max_iter=100)
        
        # All residuals should be non-negative
        assert all(res >= 0 for res in residuals)
        
        # Final residual should be below tolerance
        assert residuals[-1] < 1e-8

    def test_star_graph(self, star_graph_edges):
        """Test PageRank on a star graph (hub-and-spoke)."""
        P, v = build_matrix(star_graph_edges, alpha=0.85)
        r, _, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        # PageRank should sum to 1 and be non-negative
        assert np.isclose(r.sum(), 1.0)
        assert all(score >= 0 for score in r)
        # Peripheral nodes (1,2,3) should have similar scores (they're symmetric)
        assert np.allclose(r[1], r[2])
        assert np.allclose(r[1], r[3])

    def test_chain_graph(self):
        """Test PageRank on a chain graph."""
        edges = [(i, i+1, 1.0) for i in range(9)]  # 0 -> 1 -> 2 -> ... -> 9
        P, v = build_matrix(edges, alpha=0.85)
        r, _, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        # PageRank should sum to 1 and be non-negative
        assert np.isclose(r.sum(), 1.0)
        assert all(score >= 0 for score in r)
        # Last node is dangling and gets high PageRank from teleportation
        assert r[-1] > 0
