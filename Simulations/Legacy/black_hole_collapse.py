import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# from GraphRicciCurvature.OllivierRicci import OllivierRicci

# --- COLLAPSE PARAMETERS ---
N_NODES = 1000
ALPHA = 2.5        # Gractal critical point
BETA_NORMAL = 1.2  # Standard cohesion
BETA_BH = 5.0      # EXTREME COHESION (Black Hole)
M_LINKS = 4


def simulate_black_hole():
    print("--- SIMULATING TENSOR HUB (BH) COLLAPSE ---")
    G = nx.complete_graph(M_LINKS + 1)
    degrees = np.array([G.degree(n) for n in G.nodes()])
    nodes = np.array(G.nodes())

    for t in range(len(nodes), N_NODES):
        # Define the "Black Hole Core" (the first 20 nodes)
        # New tensors are violently attracted towards the center
        is_near_bh = (nodes < 20)
        current_beta = np.where(is_near_bh, BETA_BH, BETA_NORMAL)

        dist_causal = (t - nodes)
        dist_causal[dist_causal == 0] = 1

        # Modified DCTN formula with central attraction
        weights = (degrees ** current_beta) / (dist_causal ** ALPHA)
        probs = weights / weights.sum()

        targets = np.random.choice(nodes, size=M_LINKS, replace=False, p=probs)
        G.add_node(t)
        for target in targets:
            G.add_edge(t, target)
            degrees[target] += 1
        nodes = np.append(nodes, t)
        degrees = np.append(degrees, M_LINKS)

    # Curvature Calculation (Proxy due to missing library)
    print("Calculating metric distortion (Degree Centrality Proxy)...")
    # orc = OllivierRicci(G, alpha=0.5, method="Sinkhorn", proc=1)
    # orc.compute_ricci_curvature()
    
    # Visualization
    plt.figure(figsize=(10, 10))
    pos = nx.kamada_kawai_layout(G)
    curvs = list(nx.degree_centrality(G).values())

    nx.draw_networkx_nodes(G, pos, node_size=30, node_color=curvs, cmap=plt.cm.hot)
    nx.draw_networkx_edges(G, pos, alpha=0.05, edge_color='gray')

    plt.title("DCTN Phenomenology: Tensor Hub (Gractal Black Hole)")
    plt.axis('off')
    plt.axis('off')
    
    # Save Image
    import os
    output_path = os.path.join("../Preprints/images", 'p3_fig1_black_hole_tensor_hub.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    simulate_black_hole()
