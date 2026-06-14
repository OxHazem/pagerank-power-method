'''
graph_loader.py

Purpose:
    Read and parse graph data from various formats, with validation,
    comment/blank-line support, and conversion utilities.

Main Functions:
    - load_edge_list(path: str, directed: bool = True) -> List[Tuple[int, int]]
    - load_adjacency_list(path: str) -> Dict[int, List[int]]
    - graph_to_edge_list(graph: nx.Graph) -> List[Tuple[int, int]]
'''

import os
from typing import List, Tuple, Dict, Union
import networkx as nx
from collections import defaultdict

def graph_summary(edges: List[Tuple[int, int]])-> None :
    """
    print a summary of the graph 
    Args:
        edges: List of (source, target) integer tuples.

    Returns:
        -Nodes count
        -Edges count
        -Dangling nodes count
        -Isolated nodes count 
        -self loops 
        -min node ID
        -max node ID 
        -connected components count 
        -sample edges 


    
    """
    nodes =set()
    self_loop = 0 
    out_degree = defaultdict(int)
    in_degree  = defaultdict(int)

    dangling_nodes = 0 
    isolated_nodes = 0 

    for u,v in edges :


        nodes.add(u)
        nodes.add(v)

        out_degree[u]+=1
        in_degree[v]+=1

        if u==v : 
            self_loop+=1

    num_nodes = len(nodes)
    num_edges = len (edges)

    for node in nodes : 
        if out_degree[node]==0 :
            dangling_nodes+=1 

        if out_degree[node] == 0 and in_degree[node] == 0 :
            isolated_nodes+=1

    if nodes : 
        min_node = min(nodes)
        max_node = max(nodes)
    else : None

    graph = nx.Graph()
    graph.add_edges_from(edges)

    connected_components = nx.number_connected_components(graph)

    print("\nGraph Summary")
    print("-------------")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Dangling nodes: {dangling_nodes}")
    print(f"Isolated nodes: {isolated_nodes}")
    print(f"Self-loops: {self_loop}")
    print(f"Min node ID: {min_node}")
    print(f"Max node ID: {max_node}")
    print(f"Connected components: {connected_components}")

    print("\nSample edges:")

    for edge in edges[:5]:
        print(edge)







    
    


def load_edge_list(
    path: str,
    directed: bool = True
) -> List[Tuple[int, int]]:
    """
    Load edges from a text file where each non-comment line has two ints: u v.

    Args:
        path: File path to read.
        directed: If True, returns edges for a DiGraph; else for an undirected Graph.

    Returns:
        List of (source, target) integer tuples.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: On malformed lines.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Edge list file not found: {path}")

    edges: List[Tuple[int, int]] = []
    with open(path, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid format on line {lineno}: '{stripped}'")
            try:
                u, v = map(int, parts)
            except ValueError:
                raise ValueError(f"Non-integer node ID on line {lineno}: '{stripped}'")
            edges.append((u, v))
    return edges


def load_adjacency_list(
    path: str
) -> Dict[int, List[int]]:
    """
    Load a node-to-neighbors mapping from a text file: u v1 v2 ...

    Args:
        path: File path to read.

    Returns:
        Dict mapping node ID to list of neighbor IDs.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: On malformed lines.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Adjacency list file not found: {path}")

    adjacency: Dict[int, List[int]] = {}
    with open(path, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            try:
                node = int(parts[0])
                neighbors = [int(p) for p in parts[1:]]
            except ValueError:
                raise ValueError(f"Invalid format on line {lineno}: '{stripped}'")
            adjacency[node] = neighbors
    return adjacency


def graph_to_edge_list(
    graph: Union[nx.Graph, nx.DiGraph]
) -> List[Tuple[int, int]]:
    """
    Convert a NetworkX graph to an edge-list of integer tuples.

    Args:
        graph: A networkx.Graph or networkx.DiGraph.

    Returns:
        List of (source, target) integer tuples.

    Raises:
        TypeError: If not a NetworkX graph instance.
        ValueError: If graph contains non-integer nodes.
    """
    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise TypeError("Input must be a networkx.Graph or DiGraph.")

    edges: List[Tuple[int, int]] = []
    for u, v in graph.edges():
        try:
            u_i, v_i = int(u), int(v)
        except (ValueError, TypeError):
            raise ValueError(f"Non-integer node ID in edge: ({u}, {v})")
        edges.append((u_i, v_i))
    return edges
