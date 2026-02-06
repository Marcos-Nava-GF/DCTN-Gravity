import sys
import multiprocessing

# --- WINDOWS COMPATIBILITY PATCH ---
# Essential for GraphRicciCurvature on Windows systems
if sys.platform.startswith('win'):
    _original_get_context = multiprocessing.get_context


    def patched_get_context(method=None):
        if method == 'fork': return _original_get_context('spawn')
        return _original_get_context(method)


    multiprocessing.get_context = patched_get_context
# -----------------------------------

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.stats import linregress
import time

# --- EXPERIMENTAL CONFIGURATION ---
N_NODES = 1500  # Network size
M_LINKS = 4  # Connections per step (Coordination number)
ALPHA = 2.5  # REMOVED
GAMMA = 2.5  # Causal Decay (Critical Point: Adjust to 1.0 to see collapse)
BETA = 1.2  # Preferential Attachment Power (Mass/Gravity influence)


def generate_dctn_network(n_nodes, m_links, gamma, beta):
    """
    Generates a Gractal network using the DCTN growth rule.
    Optimized for performance using vectorization.
    """
    print(f"--- GENERATING GRACTAL NETWORK (Nodes: {n_nodes}, Gamma: {gamma}) ---")
    start_time = time.time()

    # Initialize with a complete graph nucleus
    G = nx.complete_graph(m_links + 1)

    # Use lists for mutable operations (much faster than np.append)
    degrees = [G.degree(n) for n in G.nodes()]

    # Growth Loop
    # We use a range relative to current size to avoid re-creating arrays constantly
    current_size = len(degrees)

    for t in range(current_size, n_nodes):
        # 1. Causal Distance (Vectorized)
        # Distance from new node 't' to all existing nodes [0...t-1]
        node_indices = np.arange(t)
        dist_causal = t - node_indices

        # Avoid division by zero (though conceptually distance is never 0 here)
        dist_causal = np.maximum(dist_causal, 1)

        # 2. Probability Calculation (The Gractal Equation)
        # P ~ (Degree^Beta) / (TimeDistance^Gamma)
        current_degrees = np.array(degrees)
        weights = (current_degrees ** beta) / (dist_causal ** gamma)

        weight_sum = weights.sum()
        if weight_sum == 0:
            probs = np.ones(t) / t
        else:
            probs = weights / weight_sum

        # 3. Selection & Attachment
        targets = np.random.choice(node_indices, size=m_links, replace=False, p=probs)

        G.add_node(t)
        degrees.append(m_links)  # New node starts with m_links degree

        for target in targets:
            G.add_edge(t, target)
            degrees[target] += 1

    print(f"Generation Complete. Time: {time.time() - start_time:.2f}s")
    return G

def analyze_topology(G):
    print("--- STARTING TOPOLOGICAL ANALYSIS ---")

    # 1. Ricci Curvature (Proxy)
    print("Calculating Proxy Curvature (Degree)...")
    # orc = OllivierRicci(G, alpha=0.5, method="Sinkhorn", proc=1)
    # orc.compute_ricci_curvature()
    G_ricci = G.copy() # Just copy G

    # 2. Hausdorff Dimension (Box-counting / Mass-Radius relation)
    print("Calculating Hausdorff Dimension...")
    # ... (rest same)
    sample_size = min(50, len(G.nodes()))
    seeds = np.random.choice(list(G_ricci.nodes()), size=sample_size, replace=False)

    radii_list, mass_list = [], []

    for seed in seeds:
        lengths = nx.single_source_shortest_path_length(G_ricci, seed)
        max_r = max(lengths.values())

        # Optimization: Calculate masses for all radii at once using numpy
        path_lengths = np.array(list(lengths.values()))
        for r in range(1, max_r + 1):
            mass = np.sum(path_lengths <= r)
            if mass > 1:
                radii_list.append(r)
                mass_list.append(mass)

    # Log-Log Regression to find the Fractal Dimension (slope)
    if len(radii_list) > 0:
        slope, intercept, r_val, _, _ = linregress(np.log(radii_list), np.log(mass_list))
    else:
        slope, r_val = 0, 0

    return G_ricci, slope, r_val, radii_list, mass_list


def visualize_results(G_ricci, slope, r_val, radii_list, mass_list):
    print("Generating Dual Visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- LEFT PLOT: Network Topology & Curvature ---
    pos = nx.kamada_kawai_layout(G_ricci)

    # Proxy curvature
    node_curvatures = list(nx.degree_centrality(G_ricci).values())

    avg_curv = np.mean(node_curvatures)

    # --- COLOR FIX: Mapeo Manual ---
    norm = mcolors.TwoSlopeNorm(vmin=min(node_curvatures), vcenter=avg_curv, vmax=max(node_curvatures))
    cmap = plt.cm.coolwarm

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    node_colors_explicit = [sm.to_rgba(c) for c in node_curvatures]

    nodes_plot = nx.draw_networkx_nodes(G_ricci, pos,
                                        node_size=40,
                                        node_color=node_colors_explicit,
                                        ax=ax1)

    nx.draw_networkx_edges(G_ricci, pos, alpha=0.15, edge_color="gray", ax=ax1)

    # Colorbar usando nuestro objeto manual 'sm'
    cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Scalar Ricci Curvature (Rv)', rotation=270, labelpad=15)

    ax1.set_title(f"Gractal Topology ($\\gamma={GAMMA}$)\nMean Curvature: {avg_curv:.4f}")
    ax1.axis('off')

    # --- RIGHT PLOT: Fractal Scaling ---
    ax2.scatter(np.log(radii_list), np.log(mass_list), alpha=0.2, s=15, color='black', label='Sample Data')

    # Recalculamos intercepto rápido para que la línea roja quede perfecta
    slope_recalc, intercept_recalc, _, _, _ = linregress(np.log(radii_list), np.log(mass_list))

    x_fit = np.linspace(min(np.log(radii_list)), max(np.log(radii_list)), 100)
    y_fit = slope_recalc * x_fit + intercept_recalc

    ax2.plot(x_fit, y_fit, color='red', linewidth=2, label=f'Dimension $d_H = {slope:.2f}$')

    ax2.set_xlabel("log(Radius $r$)")
    ax2.set_ylabel("log(Mass $M(r)$)")
    ax2.set_title(f"Fractal Dimension Analysis ($R^2 = {r_val ** 2:.3f}$)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.tight_layout()
    import os
    output_path = os.path.join("../Preprints/images", 'p1_fig2_hausdorff_dimension.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Image saved to {output_path}")

    print(f"\n--- FINAL REPORT ---")
    print(f"Mean Curvature: {avg_curv:.4f}")
    print(f"Hausdorff Dimension: {slope:.4f}")
    print(f"Fractal Fit (R2): {r_val ** 2:.4f}")


if __name__ == "__main__":
    # 1. Generate
    G = generate_dctn_network(N_NODES, M_LINKS, GAMMA, BETA)

    # 2. Analyze
    G_ricci, dH, r2, rads, mass = analyze_topology(G)

    # 3. Visualize
    visualize_results(G_ricci, dH, r2, rads, mass)
