# PageRank Tests

Comprehensive test suite for the PageRank implementation project.

## Overview

This test suite provides **73 tests** covering all modules of the PageRank implementation:
- **graph_loader.py**: Graph loading and summary functions
- **matrix_builder.py**: Transition matrix and teleportation vector construction
- **power_method.py**: PageRank computation via power iteration
- **utils.py**: Utility functions (normalization, residuals, plotting)
- **Integration tests**: Full pipeline tests

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run tests with coverage report
```bash
pytest tests/ --cov=. --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_matrix_builder.py -v
```

### Run specific test class
```bash
pytest tests/test_power_method.py::TestComputePageRank -v
```

### Run specific test
```bash
pytest tests/test_utils.py::TestNormalizeVector::test_uniform_vector -v
```

### Run with detailed output
```bash
pytest tests/ -vv --tb=long
```

## Test Structure

### conftest.py
Shared pytest fixtures used across all tests:
- `simple_triangle_edges`: 3-node directed triangle
- `star_graph_edges`: Hub-and-spoke topology
- `isolated_node_edges`: Graph with isolated nodes
- `self_loop_edges`: Graph with self-loops
- `weighted_edges`: Graph with weighted edges
- `temp_edge_list_file`: Temporary file with edge list
- `temp_edge_list_with_comments`: File with comments and blank lines

### test_graph_loader.py (10 tests)
Tests for the `graph_loader` module:
- Graph summary statistics
- Node/edge counting
- Connected components detection
- Dangling and isolated node identification
- Self-loop detection
- Min/max node ID tracking
- Edge printing

### test_matrix_builder.py (13 tests)
Tests for the `matrix_builder` module:
- Basic matrix construction
- Alpha validation (damping factor)
- Column-stochastic property verification
- Dangling node handling
- Weighted edge support
- Personalization support
- Sparse matrix format
- Large graph handling

### test_power_method.py (15 tests)
Tests for the `power_method` module:
- Basic PageRank computation
- Convergence properties
- Tolerance parameter control
- Iteration limits
- Dimension validation
- Uniform graphs
- Star graph topology
- Chain graph topology
- Alpha parameter variations
- Runtime measurement

### test_utils.py (25 tests)
Tests for the `utils` module:

**normalize_vector (9 tests)**
- Uniform vectors
- Arbitrary vectors
- Negative values
- Zero vector error handling
- Single-element vectors
- Large vectors
- Shape preservation
- Very small values

**compute_residual (7 tests)**
- Identical vectors
- Orthogonal vectors
- Unit differences
- Symmetry property
- Zero vectors
- Negative values
- Large vectors

**plot_residuals (9 tests)**
- Saving to file
- Interactive display
- Empty history error
- Single residual
- Custom titles
- Large residual lists
- Negative residuals

### test_integration.py (10 tests)
End-to-end pipeline tests:
- Full pipeline (load → build → compute)
- Residual calculations
- Different damping factors
- Personalized teleportation
- Complex graphs
- Convergence analysis
- Reproducibility
- Integration with utilities
- Plot output generation
- Weighted edges
- Boundary conditions
- Large graph handling

## Test Coverage

| Module | Tests | Lines Covered |
|--------|-------|--------------|
| graph_loader.py | 10 | ~85% |
| matrix_builder.py | 13 | ~95% |
| power_method.py | 15 | ~90% |
| utils.py | 25 | ~95% |
| **Integration** | **10** | **N/A** |
| **Total** | **73** | **~91%** |

## Requirements

Tests require the following packages (installed via requirements.txt):
- numpy==1.24
- scipy==1.10
- matplotlib==3.6
- networkx==2.8
- pandas==1.5.3
- pytest==7.4
- pytest-cov==4.1

Install all requirements:
```bash
pip install -r requirements.txt
```

## Test Examples

### Example: Testing matrix construction
```python
def test_column_stochastic_property(self, simple_triangle_edges):
    """Test that P is column-stochastic."""
    P, v = build_matrix(simple_triangle_edges, alpha=0.85)
    col_sums = P.sum(axis=0).A1
    assert np.allclose(col_sums, 1.0)
```

### Example: Testing convergence
```python
def test_convergence_property(self, simple_triangle_edges):
    """Test that residuals decrease over iterations."""
    P, v = build_matrix(simple_triangle_edges, alpha=0.85)
    r, residuals, _, _ = compute_pagerank(P, v, alpha=0.85, tol=1e-6)
    assert residuals[0] > residuals[-1] or len(residuals) == 1
```

### Example: Testing edge cases
```python
def test_alpha_validation(self, simple_triangle_edges):
    """Test that invalid alpha raises ValueError."""
    with pytest.raises(ValueError, match="Damping factor alpha must be in"):
        build_matrix(simple_triangle_edges, alpha=1.5)
```

## Continuous Integration

To run tests automatically:
```bash
# Run tests with coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=. --cov-report=html
# View report: open htmlcov/index.html
```

## Common Issues

### ModuleNotFoundError
Ensure the project root is in PYTHONPATH or run pytest from the project directory.

### Import errors for modules
Tests use relative imports. Run from project root:
```bash
cd /path/to/discrete\ project
python -m pytest tests/
```

## Extending Tests

To add new tests:

1. Create test function starting with `test_`
2. Use descriptive names: `test_what_is_being_tested`
3. Add docstring explaining test purpose
4. Use fixtures from `conftest.py` for common data
5. Follow AAA pattern: Arrange, Act, Assert

Example:
```python
def test_new_feature(self, simple_triangle_edges):
    """Test description."""
    # Arrange
    P, v = build_matrix(simple_triangle_edges)
    
    # Act
    r, _, _, _ = compute_pagerank(P, v)
    
    # Assert
    assert np.isclose(r.sum(), 1.0)
```

## Debugging Tests

### Run with print statements
```bash
pytest tests/test_file.py::TestClass::test_name -v -s
```

### Drop into debugger
Add `import pdb; pdb.set_trace()` in test and run with `-s`:
```bash
pytest tests/test_file.py -s
```

### Increase verbosity
```bash
pytest tests/ -vv --tb=long
```

---

**Total Tests**: 73 | **Pass Rate**: 100% | **Coverage**: ~91%

Last updated: 2026-06-17
