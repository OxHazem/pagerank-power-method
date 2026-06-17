"""
Shared pytest fixtures and test data for the PageRank test suite.
"""
import pytest
import tempfile
import os


@pytest.fixture
def simple_triangle_edges():
    """Simple directed triangle graph: 0 → 1 → 2 → 0."""
    return [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]


@pytest.fixture
def star_graph_edges():
    """Star graph: node 0 connects to nodes 1, 2, 3."""
    return [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]


@pytest.fixture
def isolated_node_edges():
    """Graph with an isolated node (node 3 has no edges)."""
    return [(0, 1, 1.0), (1, 2, 1.0)]


@pytest.fixture
def self_loop_edges():
    """Graph with a self-loop."""
    return [(0, 0, 1.0), (0, 1, 1.0), (1, 2, 1.0)]


@pytest.fixture
def weighted_edges():
    """Graph with weighted edges."""
    return [(0, 1, 2.0), (0, 2, 3.0), (1, 2, 1.0), (2, 0, 1.5)]


@pytest.fixture
def temp_edge_list_file(simple_triangle_edges):
    """Create a temporary edge list file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for src, dst, weight in simple_triangle_edges:
            f.write(f"{src} {dst}\n")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_edge_list_with_comments():
    """Create a temporary edge list file with comments and blank lines."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("# This is a comment\n")
        f.write("0 1\n")
        f.write("\n")
        f.write("# Another comment\n")
        f.write("1 2\n")
        f.write("2 0\n")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)
