import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import multiprocessing

# --- WINDOWS COMPATIBILITY PATCH ---
if sys.platform.startswith('win'):
    _original_get_context = multiprocessing.get_context
    def patched_get_context(method=None):
        if method == 'fork': return _original_get_context('spawn')
        return _original_get_context(method)
    multiprocessing.get_context = patched_get_context

# Minimal Gractal Generator extracted from gractal_generation.py
def generate_gractal(n_nodes, m_links, alpha, beta):
    G = nx.complete_graph(m_links + 1)
    degrees = [G.degree(n) for n in G.nodes()]
    current_size = len(degrees)
    for t in range(current_size, n_nodes):
        dist_causal = t - np.arange(t)
        dist_causal = np.maximum(dist_causal, 1)
        current_degrees = np.array(degrees)
        weights = (current_degrees ** beta) / (dist_causal ** alpha)
        prob = weights / weights.sum()
        targets = np.random.choice(np.arange(t), size=m_links, replace=False, p=prob)
        G.add_node(t)
        degrees.append(m_links)
        for tg in targets:
            G.add_edge(t, tg)
            degrees[tg] += 1
    return G

def plot_and_save(G, filename, title):
    plt.figure(figsize=(8, 8))
    pos = nx.kamada_kawai_layout(G) # Fruchterman is faster but Kamada is nicer for structure
    nx.draw(G, pos, node_size=20, node_color='black', alpha=0.6, edge_color='gray', width=0.5)
    plt.title(title)
    plt.axis('off')
    
    out = os.path.join("../Preprints/images", filename)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    print(f"Saved {out}")
    plt.close()

if __name__ == "__main__":
    # Generate Alpha=1.0 (Collapse)
    print("Generating alpha=1.0...")
    G1 = generate_gractal(300, 3, 1.0, 1.2) # Smaller N for visualization clarity
    plot_and_save(G1, 'p1_fig1a_topology_alpha1.png', r'Collapse Regime ($\alpha=1.0$)')

    # Generate Alpha=2.5 (Gractal)
    print("Generating alpha=2.5...")
    G2 = generate_gractal(300, 3, 2.5, 1.2)
    plot_and_save(G2, 'p1_fig1b_topology_alpha25.png', r'Gractal Regime ($\alpha=2.5$)')
