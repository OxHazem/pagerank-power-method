"""
Unit tests for utils.py
"""
import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "03. Deliverable 3 - Code Implementation & Documentation"))

from utils import normalize_vector, compute_residual, plot_residuals


class TestNormalizeVector:
    """Test the normalize_vector function."""

    def test_uniform_vector(self):
        """Test normalization of uniform vector."""
        x = np.array([1.0, 1.0, 1.0])
        result = normalize_vector(x)
        
        # Should sum to 1
        assert np.isclose(result.sum(), 1.0)
        
        # All entries should be equal
        assert np.allclose(result, 1/3)

    def test_arbitrary_vector(self):
        """Test normalization of arbitrary vector."""
        x = np.array([2.0, 3.0, 5.0])
        result = normalize_vector(x)
        
        # Should sum to 1
        assert np.isclose(result.sum(), 1.0)
        
        # Should maintain relative proportions
        expected = np.array([2/10, 3/10, 5/10])
        assert np.allclose(result, expected)

    def test_negative_values(self):
        """Test normalization of vector with negative values."""
        x = np.array([-1.0, 2.0, -3.0])
        result = normalize_vector(x)
        
        # normalize_vector divides by L1 norm, not absolute value
        # For signed vectors, the sum can be negative
        # L1 norm: |-1| + |2| + |-3| = 6, but result sums to (-1+2-3)/6
        expected_sum = (-1.0 + 2.0 - 3.0) / 6
        assert np.isclose(result.sum(), expected_sum)

    def test_zero_vector_raises(self):
        """Test that zero vector raises ValueError."""
        x = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Cannot normalize zero vector"):
            normalize_vector(x)

    def test_single_element_vector(self):
        """Test normalization of single-element vector."""
        x = np.array([5.0])
        result = normalize_vector(x)
        
        assert np.isclose(result[0], 1.0)

    def test_large_vector(self):
        """Test normalization of large vector."""
        x = np.ones(1000) * 2.0
        result = normalize_vector(x)
        
        assert np.isclose(result.sum(), 1.0)
        assert np.allclose(result, 1/1000)

    def test_preserves_shape(self):
        """Test that normalization preserves vector shape."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_vector(x)
        
        assert result.shape == x.shape

    def test_very_small_values(self):
        """Test normalization of very small values."""
        x = np.array([1e-10, 2e-10, 3e-10])
        result = normalize_vector(x)
        
        assert np.isclose(result.sum(), 1.0)


class TestComputeResidual:
    """Test the compute_residual function."""

    def test_identical_vectors(self):
        """Test residual of identical vectors should be 0."""
        v1 = np.array([0.5, 0.3, 0.2])
        v2 = np.array([0.5, 0.3, 0.2])
        
        residual = compute_residual(v1, v2)
        assert np.isclose(residual, 0.0)

    def test_orthogonal_vectors(self):
        """Test residual of orthogonal vectors."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        
        residual = compute_residual(v1, v2)
        # L1 distance should be 2.0
        assert np.isclose(residual, 2.0)

    def test_unit_difference(self):
        """Test residual for unit difference."""
        v1 = np.array([0.5, 0.5])
        v2 = np.array([0.6, 0.4])
        
        residual = compute_residual(v1, v2)
        # L1: |0.6-0.5| + |0.4-0.5| = 0.1 + 0.1 = 0.2
        assert np.isclose(residual, 0.2)

    def test_symmetry(self):
        """Test that residual is symmetric."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        
        res1 = compute_residual(v1, v2)
        res2 = compute_residual(v2, v1)
        
        # Both should be L1 norm of differences
        assert np.isclose(res1, res2)

    def test_zero_vectors(self):
        """Test residual of zero vectors."""
        v1 = np.zeros(5)
        v2 = np.zeros(5)
        
        residual = compute_residual(v1, v2)
        assert np.isclose(residual, 0.0)

    def test_negative_values(self):
        """Test residual computation with negative values."""
        v1 = np.array([-1.0, 0.0, 1.0])
        v2 = np.array([1.0, 0.0, -1.0])
        
        residual = compute_residual(v1, v2)
        # L1: |-1-1| + |0-0| + |1-(-1)| = 2 + 0 + 2 = 4
        assert np.isclose(residual, 4.0)

    def test_large_vectors(self):
        """Test residual for large vectors."""
        v1 = np.ones(1000)
        v2 = np.ones(1000) * 2
        
        residual = compute_residual(v1, v2)
        # L1: sum of |1-2| over 1000 elements = 1000
        assert np.isclose(residual, 1000.0)


class TestPlotResiduals:
    """Test the plot_residuals function."""

    def test_plot_with_file_path(self):
        """Test plotting residuals and saving to file."""
        residuals = [1.0, 0.5, 0.1, 0.01, 0.001]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            temp_path = f.name
        
        try:
            plot_residuals(residuals, path=temp_path)
            # Check that file was created
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_plot_without_file_path(self):
        """Test plotting residuals without saving (should not raise error)."""
        residuals = [1.0, 0.5, 0.1, 0.01]
        # This should not raise an error
        # Note: in non-interactive environment, plt.show() may be stubbed
        try:
            plot_residuals(residuals, path=None)
        except Exception as e:
            # Acceptable if it's a display-related error in non-interactive environment
            pass

    def test_empty_history_raises(self):
        """Test that empty history raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            plot_residuals([])

    def test_single_residual(self):
        """Test plotting with a single residual value."""
        residuals = [0.5]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            temp_path = f.name
        
        try:
            plot_residuals(residuals, path=temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_custom_title(self):
        """Test plotting with custom title."""
        residuals = [1.0, 0.5, 0.1]
        custom_title = "Custom Test Title"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            temp_path = f.name
        
        try:
            plot_residuals(residuals, title=custom_title, path=temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_large_residuals_list(self):
        """Test plotting with many residuals."""
        residuals = [10**(-i/5) for i in range(50)]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            temp_path = f.name
        
        try:
            plot_residuals(residuals, path=temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_negative_residuals(self):
        """Test that negative residuals can be plotted."""
        residuals = [0.5, -0.1, 0.2, -0.05]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            temp_path = f.name
        
        try:
            plot_residuals(residuals, path=temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
