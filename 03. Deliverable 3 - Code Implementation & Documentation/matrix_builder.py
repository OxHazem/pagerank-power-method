'''
matrix_builder.py

Purpose:
    Construct the stochastic transition matrix and teleportation vector for PageRank,
    handling dangling nodes and damping factor.

Main Function:
    build_matrix(
        edges: List[Tuple[int, int]],
        alpha: float = 0.85
    ) -> Tuple[csr_matrix, np.ndarray]

Features:
    - Infers node set from edge list.
    - Builds column-stochastic P^T (sparse) with dangling nodes as zero columns.
    - Creates uniform teleportation vector v (dense).

Usage:
    P, v = build_matrix(edges, alpha=0.85)
    # Then compute PageRank via r = alpha*P@r + (1-alpha)*v
'''

from typing import List, Tuple
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx


def build_matrix(
    edges: List[Tuple[int, int, float]],
    alpha: float = 0.85,
    personalization: int = None
) -> Tuple[csr_matrix, np.ndarray]:
    """
    Build the PageRank transition operator components:
        P: column-stochastic sparse matrix (transpose of transition matrix)
        v: uniform teleportation vector

    Args:
        edges: List of directed edges (src, dst, weight), zero-indexed.
        alpha: Damping factor in [0,1]; only validated here.

    Returns:
        P: csr_matrix, shape (n, n), columns sum to 1 for non-dangling nodes.
        v: np.ndarray, shape (n,), teleportation vector summing to 1.

    Raises:
        ValueError: If edges list is empty or alpha out of [0,1].
    """
    # Validate alpha
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Damping factor alpha must be in [0,1], got {alpha}")

    # Infer node set
    nodes = {u for u, _, _ in edges} | {v for _, v, _ in edges}
    if not nodes:
        raise ValueError("Edge list is empty; cannot infer number of nodes.")
    n = max(nodes) + 1

    # Compute out-degrees
    out_degree = np.zeros(n, dtype=float)
    for src, _, weight in edges:
        out_degree[src] += weight

    # Assemble sparse P^T
    rows, cols, data = [], [], []
    for src, dst, weight in edges:
        if out_degree[src] > 0:
            rows.append(dst)
            cols.append(src)
            data.append(weight / out_degree[src])

    P = csr_matrix((data, (rows, cols)), shape=(n, n))

    # Handle dangling nodes: columns with zero sum get uniform distribution
    dangling = np.where(out_degree == 0)[0]
    if dangling.size > 0:
        # For each dangling node j, set P[:, j] = 1/n
        for j in dangling:
            P[:, j] = np.ones(n) / n

    # Teleportation vector
    if personalization is None: 
        v = np.ones(n, dtype=float) / n
    else: 
        if personalization < 0 or personalization >= n:
            raise ValueError(f"Personalization node {personalization} is out of bounds for n={n}")
        v = np.zeros(n, dtype=float)
        v[personalization] = 1.0
        v= v / v.sum()
        

    return P, v


def build_transition_matrix(graph: nx.Graph, alpha: float = 0.85) -> csr_matrix:
    """
    Backward-compatible wrapper that builds the transition matrix from a NetworkX graph.

    Args:
        graph: Directed or undirected NetworkX graph.
        alpha: Damping factor passed through to build_matrix.

    Returns:
        Sparse transition matrix P.
    """
    edges = []
    for src, dst in graph.edges():
        weight = graph[src][dst].get("weight", 1.0)
        edges.append((src, dst, float(weight)))

    P, _ = build_matrix(edges, alpha=alpha)
    return P


def build_teleportation_vector(size: int, personalization: int = None) -> np.ndarray:
    """
    Backward-compatible wrapper that returns a teleportation vector.

    Args:
        size: Number of nodes.
        personalization: Optional personalized teleportation node.

    Returns:
        Teleportation vector of length size.
    """
    if size <= 0:
        raise ValueError(f"Graph size must be positive, got {size}")

    if personalization is None:
        return np.ones(size, dtype=float) / size

    if personalization < 0 or personalization >= size:
        raise ValueError(f"Personalization node {personalization} is out of bounds for n={size}")

    v = np.zeros(size, dtype=float)
    v[personalization] = 1.0
    return v
