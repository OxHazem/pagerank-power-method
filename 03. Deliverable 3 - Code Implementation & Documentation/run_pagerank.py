"""
run_pagerank.py

Purpose:
    Example usage script that ties together the graph loader, matrix builder,
    power method, and plotting utilities to compute and visualize PageRank.

Usage:
    python run_pagerank.py <input_path> [--format edgelist|adjlist] [--alpha ALPHA] [--tol TOL] [--max_iter MAX_ITER]
"""
import argparse
import os
import networkx as nx
import numpy as np
from graph_loader import load_edge_list, load_adjacency_list , graph_summary
from matrix_builder import build_matrix
from power_method import compute_pagerank
from utils import plot_residuals , save_pagerank_results


def main():
    parser = argparse.ArgumentParser(
        description='Compute PageRank on a graph via the Power Method'
    )
    # Input file as a positional argument
    parser.add_argument(
        'input_path',
        help='Path to graph file (edge list or adjacency list)'
    )
    parser.add_argument(
        '--format', '-f', choices=['edgelist', 'adjlist'], default='edgelist',
        help='Input file format (default: edgelist)'
    )
    parser.add_argument(
        '--alpha', '-a', type=float, default=0.85,
        help='Damping factor (default: 0.85)'
    )
    parser.add_argument(
        '--tol', '-t', type=float, default=1e-6,
        help='Convergence tolerance (L1 residual, default: 1e-6)'
    )
    parser.add_argument(
        '--max_iter', '-m', type=int, default=100,
        help='Maximum number of iterations (default: 100)'
    )
    parser.add_argument(
        '--personalize', '-p', type=int, default=None,
        help='Node id for personalize teleportation vector (default: None, uniform teleportation)'
    )
    parser.add_argument(
        '--output_format', '-of', choices=['csv', 'json'], default='csv',
        help='Format to save PageRank results (default: csv)'
    )
    parser.add_argument(
        '--output_path', '-op', type=str, default=None,
        help='Optional path to save PageRank results (CSV or JSON format inferred from extension)'
    )
    parser.add_argument(
        '--validate', '-v', action='store_true',
        help='Compare custom PageRank with NetworkX nx.pagerank()'
    )

    

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"results/pagerank_results.{args.output_format}"

    input_file = args.input_path

    if not os.path.isfile(input_file):
        parser.error(f"Input file not found: {input_file}")

    # Load edges based on chosen format
    if args.format == 'edgelist':
        edges = load_edge_list(input_file)
    else:
        adj = load_adjacency_list(input_file)
        edges = [(u, v) for u, nbrs in adj.items() for v in nbrs]


    graph_summary(edges)

    # Build transition matrix and teleport vector
    P, v = build_matrix(edges, alpha=args.alpha, personalization=args.personalize)

    # Execute Power Method
    ranks, residuals,iterations, runtime = compute_pagerank(P, v, tol=args.tol, max_iter=args.max_iter)

    if args.validate:
        n = len(ranks)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(edges)

        personalization = None
        if args.personalize is not None:
            personalization = {i: 0.0 for i in range(n)}
            personalization[args.personalize] = 1.0

        dangling = {i: 1.0 / n for i in range(n)}
        nx_scores = nx.pagerank(
            graph,
            alpha=args.alpha,
            personalization=personalization,
            dangling=dangling,
            tol=args.tol,
            max_iter=args.max_iter
        )

        nx_ranks = np.array([nx_scores[i] for i in range(n)], dtype=float)
        abs_diff = np.abs(ranks - nx_ranks)
        max_diff = float(abs_diff.max())
        mean_diff = float(abs_diff.mean())

        print("\nValidation Against NetworkX")
        print("--------------------------")
        print(f"{'Node':>6} {'Custom':>14} {'NetworkX':>14} {'AbsDiff':>14}")
        for i in range(n):
            print(f"{i:>6} {ranks[i]:>14.10f} {nx_ranks[i]:>14.10f} {abs_diff[i]:>14.10e}")

        print("\nValidation Summary")
        print("------------------")
        print(f"Max absolute difference: {max_diff:.10e}")
        print(f"Mean absolute difference: {mean_diff:.10e}")
        if max_diff < 1e-8:
            print("Validation: PASSED (custom implementation matches NetworkX)")
        else:
            print("Validation: WARNING (differences detected vs NetworkX)")

    # Output top 10 nodes by PageRank
    ranked_indices = ranks.argsort()[::-1]
    dict_ranks = {idx: ranks[idx] for idx in ranked_indices}
    print("Top nodes by PageRank:")
    for idx in ranked_indices:
        print(f"Node {idx}: {ranks[idx]:.6f}")

    # Plot convergence history
    save_pagerank_results(dict_ranks, args.output_path, format=args.output_format)
    plot_residuals(residuals)


if __name__ == '__main__':
    main()
