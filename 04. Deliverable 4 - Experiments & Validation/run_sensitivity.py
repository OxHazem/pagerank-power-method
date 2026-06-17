from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "03. Deliverable 3 - Code Implementation & Documentation"))

import networkx as nx
from matrix_builder import build_transition_matrix, build_teleportation_vector
from power_method import pagerank_power_method
from pathlib import Path

alphas = [0.60, 0.85, 0.95]
G = nx.karate_club_graph().to_directed()
P = build_transition_matrix(G)
v = build_teleportation_vector(len(G))

plots_dir = Path(__file__).parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

for alpha in alphas:
    r, residuals, iters, runtime = pagerank_power_method(P, v, alpha=alpha)
    top_5 = sorted(enumerate(r), key=lambda x: -x[1])[:5]
    top_5_nodes = [i for i, _ in top_5]
    print(f"Alpha={alpha}: {iters} iterations, {runtime:.2f}s, Top 5: {top_5_nodes}")
