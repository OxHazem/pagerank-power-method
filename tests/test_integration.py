"""
Integration tests for the complete PageRank pipeline.
"""
import pytest
import numpy as np
import sys
import tempfile
import os
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "03. Deliverable 3 - Code Implementation & Documentation"))

from graph_loader import graph_summary
from matrix_builder import build_matrix
from power_method import compute_pagerank
from utils import normalize_vector, compute_residual, plot_residuals


class TestPageRankPipeline:
    """Integration tests for the complete PageRank pipeline."""

    def test_full_pipeline_triangle(self, simple_triangle_edges):
        """Test complete pipeline: load -> build -> compute."""
        # Build matrix
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        
        # Compute PageRank
        r, residuals, iterations, runtime = compute_pagerank(
            P, v, alpha=0.85, tol=1e-6, max_iter=100
        )
        
        # Verify results
        assert r.shape == (3,)
        assert np.isclose(r.sum(), 1.0)
        assert all(score >= 0 for score in r)
        assert len(residuals) > 0
        assert iterations > 0
        assert runtime > 0

    def test_pipeline_with_residual_calculation(self):
        """Test pipeline and verify residual calculations."""
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (0, 3, 1.0), (3, 2, 1.0)]
        
        P, v = build_matrix(edges, alpha=0.85)
        r, residuals, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        # Verify that residuals make sense
        assert all(res >= 0 for res in residuals)
        assert residuals[-1] < 1e-6

    def test_pipeline_different_alphas(self, simple_triangle_edges):
        """Test pipeline with different damping factors."""
        alphas = [0.5, 0.75, 0.85, 0.95]
        results = []
        
        for alpha in alphas:
            P, v = build_matrix(simple_triangle_edges, alpha=alpha)
            r, _, _, _ = compute_pagerank(P, v, alpha=alpha, tol=1e-6, max_iter=100)
            results.append(r)
        
        # All results should be valid probability distributions
        for r in results:
            assert np.isclose(r.sum(), 1.0)
            assert all(score >= 0 for score in r)

    def test_pipeline_with_personalization(self, simple_triangle_edges):
        """Test pipeline with personalized teleportation."""
        P_uniform, v_uniform = build_matrix(simple_triangle_edges, alpha=0.85, personalization=None)
        r_uniform, _, _, _ = compute_pagerank(P_uniform, v_uniform, alpha=0.85, tol=1e-6, max_iter=100)
        
        P_personal, v_personal = build_matrix(simple_triangle_edges, alpha=0.85, personalization=0)
        r_personal, _, _, _ = compute_pagerank(P_personal, v_personal, alpha=0.85, tol=1e-6, max_iter=100)
        
        # Personalized should give different results
        assert not np.allclose(r_uniform, r_personal)
        
        # Personalized should favor node 0
        assert r_personal[0] > r_uniform[0]

    def test_pipeline_complex_graph(self):
        """Test pipeline on a more complex graph."""
        # Create a small network: 5 nodes with various connections
        edges = [
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 0, 1.0),
            (2, 4, 1.0),
            (3, 4, 1.0),
            (4, 0, 1.0),
        ]
        
        P, v = build_matrix(edges, alpha=0.85)
        r, residuals, iterations, runtime = compute_pagerank(
            P, v, alpha=0.85, tol=1e-6, max_iter=100
        )
        
        assert r.shape == (5,)
        assert np.isclose(r.sum(), 1.0)
        assert all(score >= 0 for score in r)
        assert iterations < 100  # Should converge

    def test_pipeline_convergence_analysis(self, simple_triangle_edges):
        """Test convergence properties across iterations."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        r_loose, res_loose, iter_loose, _ = compute_pagerank(
            P, v, alpha=0.85, tol=1e-3, max_iter=100
        )
        r_tight, res_tight, iter_tight, _ = compute_pagerank(
            P, v, alpha=0.85, tol=1e-8, max_iter=100
        )
        
        # Tight tolerance should require same or more iterations
        assert iter_tight >= iter_loose or np.allclose(r_loose, r_tight)

    def test_pipeline_reproducibility(self, simple_triangle_edges):
        """Test that pipeline produces reproducible results."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        r1, _, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        r2, _, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        # Results should be identical
        assert np.allclose(r1, r2)

    def test_pipeline_with_utilities(self, simple_triangle_edges):
        """Test pipeline with utility functions."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        r, residuals, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        # Test normalize_vector
        r_normalized = normalize_vector(r)
        assert np.isclose(r_normalized.sum(), 1.0)
        
        # Test compute_residual
        residual = compute_residual(r, r_normalized)
        assert residual >= 0

    def test_pipeline_output_to_plot(self, simple_triangle_edges):
        """Test pipeline with plot output."""
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        r, residuals, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        # Try to plot residuals
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            temp_path = f.name
        
        try:
            plot_residuals(residuals, path=temp_path)
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_pipeline_weighted_edges(self, weighted_edges):
        """Test pipeline with weighted edges."""
        P, v = build_matrix(weighted_edges, alpha=0.85)
        r, residuals, iterations, runtime = compute_pagerank(
            P, v, alpha=0.85, tol=1e-6, max_iter=100
        )
        
        assert r.shape == (3,)
        assert np.isclose(r.sum(), 1.0)
        assert all(score >= 0 for score in r)

    def test_graph_summary_integration(self, simple_triangle_edges, capsys):
        """Test graph_summary integration with pipeline."""
        graph_summary(simple_triangle_edges)
        captured = capsys.readouterr()
        
        assert "Graph Summary" in captured.out
        
        # Now run through the pipeline
        P, v = build_matrix(simple_triangle_edges, alpha=0.85)
        r, _, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        assert np.isclose(r.sum(), 1.0)

    def test_pipeline_boundary_conditions(self):
        """Test pipeline with boundary conditions."""
        # Single node with self-loop
        edges = [(0, 0, 1.0)]
        P, v = build_matrix(edges, alpha=0.85)
        r, _, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6, max_iter=100)
        
        assert np.isclose(r[0], 1.0)

    def test_pipeline_large_graph(self):
        """Test pipeline with a larger graph."""
        # Create a directed chain: 0 -> 1 -> 2 -> ... -> 49
        n = 50
        edges = [(i, (i+1) % n, 1.0) for i in range(n)]
        
        P, v = build_matrix(edges, alpha=0.85)
        r, residuals, iterations, runtime = compute_pagerank(
            P, v, alpha=0.85, tol=1e-6, max_iter=200
        )
        
        assert r.shape == (n,)
        assert np.isclose(r.sum(), 1.0)
        assert all(score >= 0 for score in r)
        assert iterations <= 200
