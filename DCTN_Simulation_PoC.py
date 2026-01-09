import sys
import multiprocessing

# --- WINDOWS COMPATIBILITY PATCH (CRITICAL) ---
# This must be executed BEFORE importing GraphRicciCurvature
if sys.platform.startswith('win'):
    # Save the original function
    _original_get_context = multiprocessing.get_context


    def patched_get_context(method=None):
        # If the library requests 'fork' (Linux default), force 'spawn' (Windows requirement)
        if method == 'fork':
            return _original_get_context('spawn')
        return _original_get_context(method)


    # Patch the function in the multiprocessing module
    multiprocessing.get_context = patched_get_context
# ---------------------------------------------------

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# Ensure GraphRicciCurvature is installed: pip install GraphRicciCurvature
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import time

# --- PARAMETERS ---
N_NODES = 500
M_LINKS = 4
ALPHA = 1.0  # Change to 2.5 later to observe the Gractal effect
BETA = 0.8


def run_simulation():
    print(f"--- STARTING DCTN SIMULATION (Windows Fixed) ---")
    print(f"Nodes: {N_NODES} | Alpha: {ALPHA}")

    # 1. Generate Network
    print("1. Generating causal topology...")
    G = nx.complete_graph(M_LINKS + 1)
    # Initialize degrees and nodes based on the seed graph
    degrees = np.array([G.degree(n) for n in G.nodes()])
    nodes = np.array(G.nodes())

    # Growth Algorithm
    for t in range(len(nodes), N_NODES):
        # Causal distance: time difference between current node t and existing nodes
        dist_causal = (t - nodes)
        dist_causal[dist_causal == 0] = 1  # Avoid division by zero if any

        # Interaction Probability Formula (DCTN Core)
        weights = (degrees ** BETA) / (dist_causal ** ALPHA)
        probs = weights / weights.sum()

        # Select target nodes based on probability
        targets = np.random.choice(nodes, size=M_LINKS, replace=False, p=probs)

        G.add_node(t)
        for target in targets:
            G.add_edge(t, target)
            degrees[target] += 1

        # Update lists (appending to numpy is slow for large N, but fine for N=500)
        nodes = np.append(nodes, t)
        degrees = np.append(degrees, M_LINKS)

    print(f"   -> Network generated. Total Edges: {G.number_of_edges()}")

    # 2. Calculate Curvature
    print("2. Calculating Ollivier-Ricci Curvature...")
    start_curv = time.time()

    # IMPORTANT: proc=1 guarantees maximum stability on Windows with the patch.
    # Increase proc only if you are sure the environment supports it.
    orc = OllivierRicci(G, alpha=0.5, method="Sinkhorn", proc=1, verbose="INFO")
    orc.compute_ricci_curvature()
    G_ricci = orc.G.copy()

    end_curv = time.time()
    print(f"   -> Curvature calculated in {end_curv - start_curv:.2f} seconds.")

    # 3. Extract Data
    ricci_curvatures = []
    # We calculate the average curvature of the edges connected to a node
    # to visualize it as a "node property".
    for n in G_ricci.nodes():
        if G_ricci.degree(n) > 0:
            edges = G_ricci.edges(n)
            # Access the edge curvature calculated by the library
            vals = [G_ricci[u][v]['ricciCurvature'] for u, v in edges]
            avg_k = np.mean(vals)
            ricci_curvatures.append(avg_k)
        else:
            ricci_curvatures.append(0)

    avg_global = np.mean(ricci_curvatures)
    print(f"--- RESULTS ---")
    print(f"Global Mean Curvature: {avg_global:.4f}")

    # 4. Visualization
    print("3. Generating plot...")
    plt.figure(figsize=(10, 8))

    # Use kamada_kawai for better visualization of fractal structures
    try:
        pos = nx.kamada_kawai_layout(G_ricci)
    except:
        print("   Warning: Kamada-Kawai failed, falling back to Spring layout.")
        pos = nx.spring_layout(G_ricci, seed=42)

    # Draw Nodes
    nx.draw_networkx_nodes(G_ricci, pos, node_size=50,
                           node_color=ricci_curvatures,
                           cmap=plt.cm.coolwarm,
                           alpha=0.9)
    # Draw Edges
    nx.draw_networkx_edges(G_ricci, pos, alpha=0.1, edge_color='gray')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                               norm=plt.Normalize(vmin=min(ricci_curvatures), vmax=max(ricci_curvatures)))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Ollivier-Ricci Curvature (Node Avg)')

    plt.title(f"DCTN Geometry (N={N_NODES}, Alpha={ALPHA})")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    run_simulation()
