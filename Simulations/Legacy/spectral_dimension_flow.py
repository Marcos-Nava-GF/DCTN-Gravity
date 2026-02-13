import sys
import multiprocessing

# --- WINDOWS COMPATIBILITY PATCH ---
if sys.platform.startswith('win'):
    _original_get_context = multiprocessing.get_context


    def patched_get_context(method=None):
        if method == 'fork': return _original_get_context('spawn')
        return _original_get_context(method)


    multiprocessing.get_context = patched_get_context
# -----------------------------------------

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

# Optimized Parameters (Cohesion Experiment)
N_NODES = 1500
ALPHA = 2.5
BETA = 1.2
M_LINKS = 4


def run_spectral_analysis():
    print(f"--- STARTING DCTN SPECTRAL ANALYSIS (N={N_NODES}, β={BETA}) ---")

    # 1. Cohesive Network Regeneration
    G = nx.complete_graph(M_LINKS + 1)
    degrees = np.array([G.degree(n) for n in G.nodes()])
    nodes = np.array(G.nodes())

    for t in range(len(nodes), N_NODES):
        dist_causal = (t - nodes)
        dist_causal[dist_causal == 0] = 1
        weights = (degrees ** BETA) / (dist_causal ** ALPHA)
        probs = weights / weights.sum()
        targets = np.random.choice(nodes, size=M_LINKS, replace=False, p=probs)
        G.add_node(t)
        for target in targets:
            G.add_edge(t, target)
            degrees[target] += 1
        nodes = np.append(nodes, t)
        degrees = np.append(degrees, M_LINKS)

    print(f"   -> Gractal Network generated.")

    # 2. Spectral Dimension Calculation (Random Walk)
    print("2. Calculating Return Probability and d_s Flow...")
    start_time = time.time()

    # Adjacency and transition matrix
    A = nx.adjacency_matrix(G).toarray()
    # Avoid division by zero in isolated nodes (although there shouldn't be any)
    deg_vector = np.array([d for n, d in G.degree()])
    D_inv = np.diag(1.0 / deg_vector)
    M = np.dot(D_inv, A)

    max_steps = 40
    probabilities = []
    N = len(G.nodes)
    current_M_t = np.eye(N)

    steps = np.arange(1, max_steps + 1)
    for t in steps:
        current_M_t = np.dot(current_M_t, M)
        # P(t) = Tr(M^t) / N
        return_prob = np.trace(current_M_t) / N
        probabilities.append(return_prob)

    # d_s(t) = -2 * d(log P) / d(log t)
    log_t = np.log(steps)
    log_P = np.log(probabilities)
    ds_flow = -2 * np.gradient(log_P, log_t)

    print(f"   -> Analysis completed in {time.time() - start_time:.2f} s.")

    # 3. Visualization of "Spectral Flow"
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, ds_flow, color='teal', linewidth=2, marker='o', label='Experimental d_s flow')
    plt.axhline(y=2.0, color='r', linestyle='--', label='UV Limit (Planck)')
    plt.axhline(y=4.0, color='g', linestyle='--', label='IR Target (Einstein)')
    plt.xlabel("Scale (Steps t)")
    plt.ylabel("Spectral Dimension d_s")
    plt.title("Spectral Dimension Flow (Renormalization)")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.subplot(1, 2, 2)
    plt.loglog(steps, probabilities, color='navy', label='P(t) Return Prob.')
    plt.xlabel("log(t)")
    plt.ylabel("log(P(t))")
    plt.title("Return Probability Decay")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.tight_layout()
    import os
    output_path = os.path.join("../Preprints/images", 'p1_fig3_spectral_flow.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Image saved to {output_path}")

    print(f"\n--- FINAL RESULTS ---")
    print(f"Initial d_s (Short scale): {ds_flow[0]:.4f}")
    print(f"Final d_s (Long scale): {ds_flow[-1]:.4f}")


if __name__ == "__main__":
    run_spectral_analysis()
import sys
import multiprocessing

# --- WINDOWS COMPATIBILITY PATCH ---
if sys.platform.startswith('win'):
    _original_get_context = multiprocessing.get_context


    def patched_get_context(method=None):
        if method == 'fork': return _original_get_context('spawn')
        return _original_get_context(method)


    multiprocessing.get_context = patched_get_context
# -----------------------------------------

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

# Optimized Parameters (Cohesion Experiment)
N_NODES = 1500
ALPHA = 2.5
BETA = 1.2
M_LINKS = 4


def run_spectral_analysis():
    print(f"--- STARTING DCTN SPECTRAL ANALYSIS (N={N_NODES}, β={BETA}) ---")

    # 1. Cohesive Network Regeneration
    G = nx.complete_graph(M_LINKS + 1)
    degrees = np.array([G.degree(n) for n in G.nodes()])
    nodes = np.array(G.nodes())

    for t in range(len(nodes), N_NODES):
        dist_causal = (t - nodes)
        dist_causal[dist_causal == 0] = 1
        weights = (degrees ** BETA) / (dist_causal ** ALPHA)
        probs = weights / weights.sum()
        targets = np.random.choice(nodes, size=M_LINKS, replace=False, p=probs)
        G.add_node(t)
        for target in targets:
            G.add_edge(t, target)
            degrees[target] += 1
        nodes = np.append(nodes, t)
        degrees = np.append(degrees, M_LINKS)

    print(f"   -> Gractal Network generated.")

    # 2. Spectral Dimension Calculation (Random Walk)
    print("2. Calculating Return Probability and d_s Flow...")
    start_time = time.time()

    # Adjacency and transition matrix
    A = nx.adjacency_matrix(G).toarray()
    # Avoid division by zero in isolated nodes (although there shouldn't be any)
    deg_vector = np.array([d for n, d in G.degree()])
    D_inv = np.diag(1.0 / deg_vector)
    M = np.dot(D_inv, A)

    max_steps = 40
    probabilities = []
    N = len(G.nodes)
    current_M_t = np.eye(N)

    steps = np.arange(1, max_steps + 1)
    for t in steps:
        current_M_t = np.dot(current_M_t, M)
        # P(t) = Tr(M^t) / N
        return_prob = np.trace(current_M_t) / N
        probabilities.append(return_prob)

    # d_s(t) = -2 * d(log P) / d(log t)
    log_t = np.log(steps)
    log_P = np.log(probabilities)
    ds_flow = -2 * np.gradient(log_P, log_t)

    print(f"   -> Analysis completed in {time.time() - start_time:.2f} s.")

    # 3. Visualization of "Spectral Flow"
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, ds_flow, color='teal', linewidth=2, marker='o', label='Experimental d_s flow')
    plt.axhline(y=2.0, color='r', linestyle='--', label='UV Limit (Planck)')
    plt.axhline(y=4.0, color='g', linestyle='--', label='IR Target (Einstein)')
    plt.xlabel("Scale (Steps t)")
    plt.ylabel("Spectral Dimension d_s")
    plt.title("Spectral Dimension Flow (Renormalization)")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.subplot(1, 2, 2)
    plt.loglog(steps, probabilities, color='navy', label='P(t) Return Prob.')
    plt.xlabel("log(t)")
    plt.ylabel("log(P(t))")
    plt.title("Return Probability Decay")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.tight_layout()
    import os
    output_path = os.path.join("../Preprints/images", 'p1_fig3_spectral_flow.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Image saved to {output_path}")

    print(f"\n--- FINAL RESULTS ---")
    print(f"Initial d_s (Short scale): {ds_flow[0]:.4f}")
    print(f"Final d_s (Long scale): {ds_flow[-1]:.4f}")


if __name__ == "__main__":
    run_spectral_analysis()
